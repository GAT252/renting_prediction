import os
import re
import json
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from scraper import scrape_and_train
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_here_2'

@app.route('/')
def select_page():
    """地域名入力ページを表示"""
    return render_template('select_region.html')

@app.route('/train', methods=['POST'])
def train_page():
    """
    入力された地域名でモデルの作成または読み込みを行い、
    予測入力フォームに遷移する
    """
    region_name = request.form.get('region_name')
    if not region_name:
        flash('地域名を入力してください。')
        return redirect(url_for('select_page'))

    # ファイル名として使えるようにサニタイズ
    safe_region_name = re.sub(r'[\\/*?:"<>|]', "", region_name)
    model_path = f'models/{safe_region_name}_model.pkl'
    columns_path = f'models/{safe_region_name}_columns.json'
    madori_list = []
    try:
        if not os.path.exists(model_path):
            flash(f'「{region_name}」の学習モデルを作成します。数分かかる場合があります...')
            madori_list = scrape_and_train(region_name)
        else:
            flash(f'「{region_name}」の学習済みモデルを読み込みました。')
        
            madori_list_path = f'models/{safe_region_name}_madori_list.json'
            
            print(f"Attempting to load madori list from: {madori_list_path}")

            if os.path.exists(madori_list_path):
                with open(madori_list_path, 'r', encoding='utf-8') as f:
                    madori_list = json.load(f)
            
                print(f"Madori list loaded from file: {madori_list}")
            else:
                
                print(f"ERROR: Madori list file not found at {madori_list_path}")
            
        

        # 最終的にリストが空だった場合に警告を出す
        if not madori_list:
            print("WARNING: Madori list is empty before rendering template.")

        return render_template('index.html', 
                               region_name=region_name, 
                               safe_region_name=safe_region_name,
                               madori_list=madori_list)
    except Exception as e:
        traceback.print_exc() # トレースバックを出力
        flash(f'エラーが発生しました: {e}')
        return redirect(url_for('select_page'))

@app.route('/predict', methods=['POST'])
def predict_page():
    """フォーム入力値から家賃を予測する"""
    try:
        region_name = request.form.get('region_name')
        safe_region_name = request.form.get('safe_region_name')
        menseki = float(request.form.get('menseki'))
        year = int(request.form.get('year'))
        minute = int(request.form.get('minute'))
        selected_madori = request.form.get('madori')

        model = joblib.load(f'models/{safe_region_name}_model.pkl')
        with open(f'models/{safe_region_name}_columns.json', 'r', encoding='utf-8') as f:
            columns = json.load(f)
        #予測データフレームの作成
        pred_df = pd.DataFrame(columns=columns)
        pred_df.loc[0, :] = 0
        
        pred_df['専有面積'] = menseki
        pred_df['築年数'] = year
        pred_df['徒歩'] = minute
        
        selected_madori_col = f"madori_{selected_madori}"
        if selected_madori_col in pred_df.columns:
            pred_df[selected_madori_col] = 1
        
        prediction = model.predict(pred_df)
        fee = round(prediction[0], 2)
        
        return render_template('result.html', fee=fee, region_name=region_name)

    except (ValueError, FileNotFoundError) as e:
        traceback.print_exc() # トレースバックをコンソールに出力
        flash(f'予測中にエラーが発生しました: {e}')
        return redirect(url_for('select_page'))

if __name__ == '__main__':
    app.run(debug=True)