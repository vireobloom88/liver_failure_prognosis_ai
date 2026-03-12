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
rf_model = joblib.load(os.path.join(MODEL_DIR, "final_best_rf_model_8_data_without_imputation_production_model.joblib"))
dt_model = joblib.load(os.path.join(MODEL_DIR, "final_best_dt_model_8_data_without_imputation_production_model.joblib"))
ebm_model = joblib.load(os.path.join(MODEL_DIR, "final_best_ebm_model_8_data_without_imputation_production_model.joblib"))

cb_model = CatBoostClassifier()
cb_model.load_model(os.path.join(MODEL_DIR, "final_best_cb_model_8_data_without_imputation_production_model.cbm"))


# --- カテゴリ変換（肝萎縮・DIC・腹水 共通） ---
def convert_category(value):
    mapping = {"無": 2, "有": 1}
    return mapping.get(value, 0)


# --- 予測ラベル変換 ---
def convert_label(x):
    return "死亡予測" if x == 1 else "生存予測"


# --- 予測関数（8特徴量版） ---
def predict_all_models(kaniishuku, tb, comp146, alt, dic, ascites, rbc, inr):

    # カテゴリ変換
    kaniishuku_num = convert_category(kaniishuku)
    dic_num = convert_category(dic)
    ascites_num = convert_category(ascites)

    # DataFrame（モデルの特徴量順に合わせる）
    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": tb,
        "１４６合併症数": comp146,
        "ＡＬＴ": alt,
        "DIC": dic_num,
        "腹水": ascites_num,
        "赤血球": rbc,
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

    importances = {}
    return results, importances

def explain_with_shap(model, input_df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, 8, 2)

    shap_raw = shap_values[0]  # shape = (8, 2)

    # 生存（クラス0）の SHAP のみ使用
    shap_0 = shap_raw[:, 0]

    # ndarray → float に変換
    shap_0 = [float(v) for v in shap_0]

    # ★ 8特徴量に更新 ★
    features = [
        "肝萎縮", "ＴＢ", "１４６合併症数", "ＡＬＴ",
        "DIC", "腹水", "赤血球", "ＩＮＲ"
    ]

    return {
        "features": features,
        "shap_0": shap_0
    }
def explain_with_shap_catboost(model, input_df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, 8)

    # CatBoost はクラス1（死亡）の SHAP
    shap_1 = shap_values[0]
    shap_0 = -shap_1  # 生存SHAP = -死亡SHAP

    shap_0 = [float(v) for v in shap_0]

    # ★ 8特徴量に更新 ★
    features = [
        "肝萎縮", "ＴＢ", "１４６合併症数", "ＡＬＴ",
        "DIC", "腹水", "赤血球", "ＩＮＲ"
    ]

    return {
        "features": features,
        "shap_0": shap_0
    }

def explain_with_shap_dt(model, input_df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)  # shape = (1, 8, 2)

    shap_raw = shap_values[0]  # shape = (8, 2)

    shap_0 = shap_raw[:, 0]
    shap_0 = [float(v) for v in shap_0]

    # ★ 8特徴量に更新 ★
    features = [
        "肝萎縮", "ＴＢ", "１４６合併症数", "ＡＬＴ",
        "DIC", "腹水", "赤血球", "ＩＮＲ"
    ]

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
        n = n.replace("１４６合併症数", "１４６合併症数")  # そのまま
        cleaned_names.append(n)

    return {
        "features": cleaned_names,
        "shap_0": scores
    }