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
    convert_category   # ← convert_kaniishuku ではなく統一
)
import pandas as pd
import json


def index(request):
    return render(request, "myapp/index.html")


# -----------------------------
#   SHAP（ランダムフォレスト）
# -----------------------------
def rf_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    input_df = pd.DataFrame([{
        "肝萎縮": convert_category(data["kaniishuku"]),
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "ＡＬＴ": data["alt"],
        "DIC": convert_category(data["dic"]),
        "腹水": convert_category(data["ascites"]),
        "赤血球": data["rbc"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap(rf_model, input_df)

    return render(request, "myapp/rf_importance.html", {
        "shap_json": json.dumps(shap_result)
    })


# -----------------------------
#   SHAP（EBM）
# -----------------------------
def ebm_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    input_df = pd.DataFrame([{
        "肝萎縮": convert_category(data["kaniishuku"]),
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "ＡＬＴ": data["alt"],
        "DIC": convert_category(data["dic"]),
        "腹水": convert_category(data["ascites"]),
        "赤血球": data["rbc"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_ebm(ebm_model, input_df)

    return render(request, "myapp/ebm_importance.html", {
        "shap_json": json.dumps(shap_result)
    })


# -----------------------------
#   SHAP（CatBoost）
# -----------------------------
def cb_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    input_df = pd.DataFrame([{
        "肝萎縮": convert_category(data["kaniishuku"]),
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "ＡＬＴ": data["alt"],
        "DIC": convert_category(data["dic"]),
        "腹水": convert_category(data["ascites"]),
        "赤血球": data["rbc"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap_catboost(cb_model, input_df)

    return render(request, "myapp/cb_importance.html", {
        "shap_json": json.dumps(shap_result)
    })


# -----------------------------
#   SHAP（決定木）
# -----------------------------
def dt_importance(request):

    if "last_input" not in request.session:
        return redirect("index")

    data = request.session["last_input"]

    input_df = pd.DataFrame([{
        "肝萎縮": convert_category(data["kaniishuku"]),
        "ＴＢ": data["tb"],
        "１４６合併症数": data["comp146"],
        "ＡＬＴ": data["alt"],
        "DIC": convert_category(data["dic"]),
        "腹水": convert_category(data["ascites"]),
        "赤血球": data["rbc"],
        "ＩＮＲ": data["inr"]
    }])

    shap_result = explain_with_shap_dt(dt_model, input_df)

    return render(request, "myapp/dt_importance.html", {
        "shap_json": json.dumps(shap_result)
    })


# -----------------------------
#   予測ビュー
# -----------------------------
def predict_view(request):

    # --- POST（初回予測） ---
    if request.method == "POST":

        tb = request.POST.get("tb")
        atrophy = request.POST.get("atrophy")
        comp146 = request.POST.get("complication")
        inr = request.POST.get("inr")
        alt = request.POST.get("alt")
        dic = request.POST.get("dic")
        ascites = request.POST.get("ascites")
        rbc = request.POST.get("rbc")

        if None in [tb, atrophy, comp146, inr, alt, dic, ascites, rbc]:
            return render(request, "myapp/result.html", {
                "error": "入力が不足しています。全ての項目を入力してください。"
            })

        tb = float(tb)
        comp146 = int(float(comp146))
        inr = float(inr)
        alt = float(alt)
        rbc = float(rbc)

        # 入力画面 → 日本語に変換
        mapping = {"no": "無", "yes": "有"}

        kaniishuku = mapping.get(atrophy, "無")
        dic_jp = mapping.get(dic, "無")
        ascites_jp = mapping.get(ascites, "無")

        # ★ セッション保存（日本語のカテゴリを保存）
        request.session["last_input"] = {
            "tb": tb,
            "comp146": comp146,
            "inr": inr,
            "alt": alt,
            "rbc": rbc,
            "kaniishuku": kaniishuku,
            "dic": dic_jp,
            "ascites": ascites_jp
        }

    # --- GET（寄与度ページから戻る） ---
    else:
        if "last_input" not in request.session:
            return redirect("index")

        data = request.session["last_input"]

        tb = data["tb"]
        comp146 = data["comp146"]
        inr = data["inr"]
        alt = data["alt"]
        rbc = data["rbc"]
        kaniishuku = data["kaniishuku"]
        dic_jp = data["dic"]
        ascites_jp = data["ascites"]

    # --- 予測実行（8特徴量） ---
    results, importances = predict_all_models(
        kaniishuku, tb, comp146, alt, dic_jp, ascites_jp, rbc, inr
    )

    return render(request, "myapp/result.html", {
        "results": results,
        "importances": importances
    })