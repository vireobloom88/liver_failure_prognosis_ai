from django.shortcuts import render

def index(request):
    return render(request, "myapp/index.html")

# myapp/views.py

from django.shortcuts import render
from .predict_models import predict_all_models


def predict_view(request):
    if request.method == "POST":
        # index.html の name に合わせて取得
        age = request.POST.get("age")
        tb = request.POST.get("tb")
        atrophy = request.POST.get("atrophy")  # 無/有/不明
        comp146 = request.POST.get("complication")
        inr = request.POST.get("inr")

        # 入力チェック
        if None in [age, tb, atrophy, comp146, inr]:
            return render(request, "myapp/result.html", {
                "error": "入力が不足しています。全ての項目を入力してください。"
            })

        # 型変換
        age = int(age)
        tb = float(tb)
        comp146 = int(comp146)
        inr = float(inr)

        # 肝萎縮の値を日本語に変換（predict_models.py の仕様に合わせる）
        mapping = {
            "no": "無",
            "yes": "有",
            "unknown": "不明"
        }
        kaniishuku = mapping.get(atrophy, "不明")

        # 予測
        results, importances = predict_all_models(kaniishuku, tb, comp146, age, inr)

        return render(request, "myapp/result.html", {
            "results": results,
            "importances": importances
        })

    return render(request, "myapp/index.html")
