from django.shortcuts import render, redirect
from .predict_models import (
    predict_all_models,
    explain_with_shap,
    explain_with_shap_dt,
    explain_with_shap_catboost,
    explain_with_ebm,
    rf_model,
    dt_model,
    cb_model,
    ebm_model,
    convert_kaniishuku
)
import pandas as pd
import json

def index(request):
    return render(request, "myapp/index.html")




def rf_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    kaniishuku_num = convert_kaniishuku(data["kaniishuku"])

    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "年齢": data["age"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap(rf_model, input_df)

    return render(request, "myapp/rf_importance.html", {
        "shap_json": json.dumps(shap_result)
    })


def ebm_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    kaniishuku_num = convert_kaniishuku(data["kaniishuku"])

    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "年齢": data["age"],
        "ＩＮＲ": data["inr"]
    }])

    # ★ ここで EBM の寄与度関数を呼び出す
    shap_result = explain_with_ebm(ebm_model, input_df)

    return render(request, "myapp/ebm_importance.html", {
        "shap_json": json.dumps(shap_result)
    })
def cb_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    kaniishuku_num = convert_kaniishuku(data["kaniishuku"])

    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "年齢": data["age"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap_catboost(cb_model, input_df)

    return render(request, "myapp/cb_importance.html", {
        "shap_json": json.dumps(shap_result)
    })

def dt_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    kaniishuku_num = convert_kaniishuku(data["kaniishuku"])

    input_df = pd.DataFrame([{
        "肝萎縮": kaniishuku_num,
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "年齢": data["age"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap_dt(dt_model, input_df)

    return render(request, "myapp/dt_importance.html", {
        "shap_json": json.dumps(shap_result)
    })



def predict_view(request):

    # --- POST（初回予測） ---
    if request.method == "POST":
        age = request.POST.get("age")
        tb = request.POST.get("tb")
        atrophy = request.POST.get("atrophy")
        comp146 = request.POST.get("complication")
        inr = request.POST.get("inr")

        if None in [age, tb, atrophy, comp146, inr]:
            return render(request, "myapp/result.html", {
                "error": "入力が不足しています。全ての項目を入力してください。"
            })

        age = int(age)
        tb = float(tb)
        comp146 = int(comp146)
        inr = float(inr)

        # 入力画面 → 日本語に変換
        mapping = {
            "no": "無",
            "yes": "有",
            "unknown": "不明"
        }
        kaniishuku = mapping.get(atrophy, "不明")

        # ★ セッションには「日本語の肝萎縮」を保存（数値は保存しない）
        request.session["last_input"] = {
            "age": age,
            "tb": tb,
            "comp146": comp146,
            "inr": inr,
            "kaniishuku": kaniishuku
        }

    # --- GET（特徴量寄与度ページから戻る） ---
    else:
        if "last_input" not in request.session:
            return redirect("index")

        data = request.session["last_input"]

        # ★ POST と同じ形式で復元（日本語の肝萎縮）
        age = data["age"]
        tb = data["tb"]
        comp146 = data["comp146"]
        inr = data["inr"]
        kaniishuku = data["kaniishuku"]

    # --- 予測実行（POST でも GET でもここに来る） ---
    results, importances = predict_all_models(
        kaniishuku, tb, comp146, age, inr
    )

    return render(request, "myapp/result.html", {
        "results": results,
        "importances": importances
    })