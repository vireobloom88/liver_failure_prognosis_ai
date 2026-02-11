# myapp/predict_models.py

import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from django.conf import settings
import shap
import numpy as np


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

    # --- 結果 ---
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

    # ★ SHAP/LIME を使うので importances は空でOK
    importances = {}

    return results, importances

def explain_with_shap(model, input_df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, 5, 2)

    shap_raw = shap_values[0]  # shape = (5, 2)

    # 生存（クラス0）の SHAP のみ使用
    shap_0 = shap_raw[:, 0]

    # ndarray → float に変換
    shap_0 = [float(v) for v in shap_0]

    # 特徴量名を修正（合併症数に変更）
    features = list(input_df.columns)
    features = ["肝萎縮", "ＴＢ", "合併症数", "年齢", "ＩＮＲ"]

    return {
        "features": features,
        "shap_0": shap_0
    }
def explain_with_shap_catboost(model, input_df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, n_features)

   

    # CatBoost は (1, n_features) の場合 → クラス1（死亡）の SHAP
    shap_1 = shap_values[0]  # shape = (n_features,)
    shap_0 = -shap_1         # 生存SHAP = -死亡SHAP

    shap_0 = [float(v) for v in shap_0]

    features = ["肝萎縮", "ＴＢ", "合併症数", "年齢", "ＩＮＲ"]

    return {
        "features": features,
        "shap_0": shap_0
    }
def explain_with_shap_dt(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, n_features, 2)
   

    shap_raw = shap_values[0]  # shape = (n_features, 2)

    shap_0 = shap_raw[:, 0]

    shap_0 = [float(v) for v in shap_0]

    features = ["肝萎縮", "ＴＢ", "合併症数", "年齢", "ＩＮＲ"]

    return {
        "features": features,
        "shap_0": shap_0
    }
def explain_with_ebm(model, input_df):

    explanation = model.explain_local(input_df)
    internal = explanation._internal_obj

    specific = internal["specific"][0]

    names = specific["names"]
    scores = specific["scores"]

    # numpy.float64 → float
    scores = [float(v) for v in scores]

    # ★★★ 符号を反転（これが重要） ★★★
    scores = [-v for v in scores]

    # 特徴量名の整形
    cleaned_names = []
    for n in names:
        n = n.replace("１４６合併症数", "合併症数")
        cleaned_names.append(n)

    return {
        "features": cleaned_names,
        "shap_0": scores
    }