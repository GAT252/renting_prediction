import os
import re
import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

# モデルとカラム情報を保存するディレクトリを作成
if not os.path.exists('models'):
    os.makedirs('models')

def scrape_and_train(region_name):
    print(f"Scraping and training for region: {region_name}...")
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # 各ページで取得したDataFrameを格納するリスト
    all_dfs = []
    
    try:
        # SUUMOにアクセスして地域名で検索
        driver.get("https://suumo.jp/chintai/kanto/")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'kwd')))
        
        text_box = driver.find_element(By.NAME, 'kwd')
        text_box.send_keys(region_name)
        btn = driver.find_element(By.ID, 'js-footerKensakuBtn') # 元のスクリプトで使われていたボタンID
        btn.click()
        
        page_num = 1
        MAX_PAGES = 5  # スクレイピングする最大ページ数を設定

        while page_num <= MAX_PAGES:
            print(f"Scraping page {page_num}...")
            # ページが読み込まれるのを待つ
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "infodatabox-details-txt"))
            )
            
            # 賃料、共益費、礼金、専有面積、間取りの取得
            el = driver.find_elements(By.CLASS_NAME, "infodatabox-details-txt")
            texts = [e.text for e in el]
            if not texts:
                print("No properties found on this page.")
                break
            rows = [texts[i:i+5] for i in range(0, len(texts), 5)]
            df_base = pd.DataFrame(rows, columns=["賃料", "共益費", "礼金", "専有面積", "間取り"])

            # 築年数と徒歩の取得
            results_year = []
            results_walk = []
            
            # ページ上の物件数に合わせてループ
            for idx in range(len(df_base)):
                ul_i = (idx // 5) + 1
                li_j = (idx % 5) + 1
                
                # 築年数
                try:
                    xpath_year = f'//*[@id="js-bukkenList"]/form/ul[{ul_i}]/li[{li_j}]/div[2]/div[2]/div[1]/div/div[2]/div/div[2]/table/tbody/tr[2]/td[3]/div'
                    element = driver.find_element(By.XPATH, xpath_year)
                    results_year.append(element.text)
                except Exception:
                    results_year.append("取得失敗")
            
                # 徒歩
                try:
                    xpath_walk = f'//*[@id="js-bukkenList"]/form/ul[{ul_i}]/li[{li_j}]/div[2]/div[2]/div[1]/div/div[2]/div/div[1]/table/tbody/tr[2]/td[3]/div'
                    element = driver.find_element(By.XPATH, xpath_walk)
                    results_walk.append(element.text)
                except Exception:
                    results_walk.append("取得失敗")

            s1 = pd.Series(results_year, name="築年数")
            s2 = pd.Series(results_walk, name="徒歩")
            
            # ページ内のデータを結合し、リストに追加
            df_page = pd.concat([df_base, s1, s2], axis=1)
            all_dfs.append(df_page)

            # 「次へ」ボタンを探してクリック
            try:
                next_button = driver.find_element(By.XPATH, '//p[@class="pagination-parts"]/a[text()="次へ"]')
                driver.execute_script("arguments[0].click();", next_button)
                page_num += 1
                time.sleep(2)
            except NoSuchElementException:
                print("Last page reached. Finishing scrape.")
                break
                
    finally:
        driver.quit()

    if not all_dfs:
        print("No property data was scraped.")
        return []

    # 全ページのDataFrameを一つに結合
    df = pd.concat(all_dfs, ignore_index=True)

    #データ前処理 
    print("Data processing...")
    
    # 取得に失敗した行や、不要な列を削除
    df = df[df['築年数'] != '取得失敗'].copy()
    df = df[df['徒歩'] != '取得失敗'].copy()
    df.drop(columns=["共益費", "礼金"], inplace=True)
    
    # 値の整形
    df = df.applymap(lambda x: re.sub('\n', ' ', x) if isinstance(x, str) else x)
    df['賃料'] = df['賃料'].str.replace('万円', '').astype(float)
    df['専有面積'] = df['専有面積'].str.replace('m2', '').astype(float)
    df['築年数'] = df['築年数'].str.replace('新', '0').str.replace('築', '').str.replace('年', '').astype(int)
    df['徒歩'] = df['徒歩'].str.extract(r'(\d+)').astype(int)
    
    original_madori_list = sorted(df['間取り'].unique().tolist())
    df = pd.get_dummies(df, columns=['間取り'], drop_first=True)

    print(f"Data processing finished. Total {len(df)} properties found.")

    # モデルの学習
    print("\nStarting model training...")

    X = df.drop('賃料', axis=1)
    y = df['賃料']
    
    if X.empty or y.empty:
        print("Not enough data to train the model.")
        return []

    safe_region_name = re.sub(r'[\\/*?:"<>|]', "", region_name)

    columns_to_save = X.columns.tolist()
    with open(f'models/{safe_region_name}_columns.json', 'w', encoding='utf-8') as f:
        json.dump(columns_to_save, f, ensure_ascii=False, indent=4)
        
    madori_list_path = f'models/{safe_region_name}_madori_list.json'
    with open(madori_list_path, 'w', encoding='utf-8') as f:
        json.dump(original_madori_list, f, ensure_ascii=False, indent=4)
        
    x_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = MLPRegressor(solver="adam", random_state=0, max_iter=5000, hidden_layer_sizes=(100, 50))
    model.fit(x_train, y_train)

    joblib.dump(model, f'models/{safe_region_name}_model.pkl', compress=3)
    print(f"Model for {safe_region_name} has been saved.")

    return original_madori_list