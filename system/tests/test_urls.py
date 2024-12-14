from django.test import SimpleTestCase
from django.urls import resolve, reverse
from myapp.views import available_tickers, train_model, get_sentiment_analysis

# Ensure all urls are correctly linked and deployed
class TestUrls(SimpleTestCase):
    def test_available_tickers_url(self):
        url = reverse('available-tickers')
        self.assertEqual(resolve(url).func, available_tickers)

    def test_train_model_url(self):
        url = reverse('train-model')
        self.assertEqual(resolve(url).func, train_model)

    def test_get_sentiment_analysis_url(self):
        url = reverse('get-sentiment-analysis')
        self.assertEqual(resolve(url).func, get_sentiment_analysis)