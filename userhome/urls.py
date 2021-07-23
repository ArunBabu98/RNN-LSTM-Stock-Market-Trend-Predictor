
from django.urls import path
from . import views


urlpatterns = [
    path('', views.userhome, name='userhome'),
    path('scrape', views.scrape, name='scrape'),
    path('predict', views.predict, name='predict'),
]