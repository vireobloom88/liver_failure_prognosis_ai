from django.contrib import admin
from django.urls import path
from myapp.views import (
    index,
    predict_view,
    rf_importance,
    ebm_importance,
    cb_importance,
    dt_importance,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('predict/', predict_view, name='predict'),

    # --- 特徴量寄与度ページ ---
    path('importance/rf/', rf_importance, name='rf_importance'),
    path('importance/ebm/', ebm_importance, name='ebm_importance'),
    path('importance/cb/', cb_importance, name='cb_importance'),
    path('importance/dt/', dt_importance, name='dt_importance'),
]