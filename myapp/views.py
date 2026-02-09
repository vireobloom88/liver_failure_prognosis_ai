from django.shortcuts import render

def index(request):
    return render(request, "myapp/index.html")


def predict(request):
    if request.method == "POST":
        age = int(request.POST["age"])
        tb = float(request.POST["tb"])
        atrophy = request.POST["atrophy"]
        complication = int(request.POST["complication"])
        inr = float(request.POST["inr"])

        # --- ここで AI モデルを動かす ---
        # 例としてダミーの結果を返す
        results = {
            "モデルA": {"prediction": "生存", "probability": 92.1},
            "モデルB": {"prediction": "生存", "probability": 88.4},
            "モデルC": {"prediction": "死亡", "probability": 63.2},
            "モデルD": {"prediction": "生存", "probability": 75.0},
        }

        importances = {
            "モデルA": {"Age": 0.12, "TB": 0.33, "INR": 0.55},
            "モデルB": {"Age": 0.22, "TB": 0.41, "INR": 0.37},
            "モデルC": {"Age": 0.18, "TB": 0.29, "INR": 0.53},
            "モデルD": {"Age": 0.10, "TB": 0.45, "INR": 0.45},
        }

        return render(request, "myapp/result.html", {
            "results": results,
            "importances": importances,
        })