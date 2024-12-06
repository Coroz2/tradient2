from django.urls import path
from . import views

urlpatterns = [
    path('available-tickers/', views.available_tickers, name='available-tickers'),
    path('train-model/', views.train_model, name='train-model'),
    path('get-sentiment-analysis/', views.get_sentiment_analysis, name='get_sentiment_analysis'),

]