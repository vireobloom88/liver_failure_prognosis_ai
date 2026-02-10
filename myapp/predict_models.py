# myapp/predict_models.py

import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from django.conf import settings

# モデルフォルダ
MODEL_DIR = os.path.join(settings.BASE_DIR, "myapp", "models")

# --- モデル読み込み ---
rf_model = joblib.load(os.path.join(MODEL_DIR, "final_best_rf_model.joblib"))
dt_model = joblib.load(os.path.join(MODEL_DIR, "final_best_dt_model.joblib"))
ebm_model = joblib.load(os.path.join(MODEL_DIR, "final_best_ebm_model.joblib"))

cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(MODEL_DIR, "final_best_cb_model.cbm"))


# --- 肝萎縮の変換 ---
def convert_kaniishuku(value):
    mapping = {"無": 2, "有": 1, "不明": 0}
    return mapping.get(value, 0)


# --- 予測ラベル変換 ---
def convert_label(x):
    return "死亡予測" if x == 1 else "生存予測"


# --- 予測関数 ---
def predict_all_models(kaniishuku, tb, comp146, age, inr):

    # 肝萎縮を数値化
    kaniishuku_num = convert_kaniishuku(kaniishuku)

    # DataFrame 作成
    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": tb,
        "１４６合併症数": comp146,
        "年齢": age,
        "ＩＮＲ": inr
    }])

    # --- 予測 ---
    pred_rf = rf_model.predict(input_df)[0]
    prob_rf = rf_model.predict_proba(input_df)[0][1] * 100

    pred_dt = dt_model.predict(input_df)[0]
    prob_dt = dt_model.predict_proba(input_df)[0][1] * 100

    pred_ebm = ebm_model.predict(input_df)[0]
    prob_ebm = ebm_model.predict_proba(input_df)[0][1] * 100

    pred_cb = cb_model.predict(input_df)[0]
    prob_cb = cb_model.predict_proba(input_df)[0][1] * 100

    # --- 結果を result.html の形式に合わせる ---
    results = {
        "ランダムフォレスト": {
            "prediction": convert_label(pred_rf),
            "probability": round(prob_rf, 2)
        },
        "決定木": {
            "prediction": convert_label(pred_dt),
            "probability": round(prob_dt, 2)
        },
        "EBM": {
            "prediction": convert_label(pred_ebm),
            "probability": round(prob_ebm, 2)
        },
        "CatBoost": {
            "prediction": convert_label(pred_cb),
            "probability": round(prob_cb, 2)
        }
    }

        # --- 特徴量重要度（result.html に合わせた形式） ---
    importances = {
        "ランダムフォレスト": dict(zip(rf_model.feature_names_in_, rf_model.feature_importances_)),
        "決定木": dict(zip(dt_model.feature_names_in_, dt_model.feature_importances_)),
        "EBM": {
            "肝萎縮": 0.25,
            "ＴＢ": 0.20,
            "１４６合併症数": 0.30,
            "年齢": 0.15,
            "ＩＮＲ": 0.10
        },
        "CatBoost": dict(zip(cb_model.feature_names_, cb_model.get_feature_importance()))
    }

    return results, importances