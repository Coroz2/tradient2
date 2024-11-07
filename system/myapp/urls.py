from django.urls import path
from . import views

urlpatterns = [
    path('train-model/', views.train_model, name='train-model'),
    path('available-tickers/', views.available_tickers, name='available-tickers'),
]