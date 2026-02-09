from django.contrib import admin
from django.urls import path
from myapp.views import index, predict

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('predict/', predict, name='predict'),
]