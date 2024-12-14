from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from django.conf import settings
from django.db import connections

class TestTrainModel(TestCase):

    # Testing available_tickers
    # Ensure response returns a list and includes APPL as one of its element
    def test_available_tickers(self):
        response = self.client.get(reverse('available-tickers'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.json(), list)
        self.assertIn({"ticker": "AAPL", "company name": "Apple Inc.", "exchange": "Nasdaq Global Select"}, response.json())

    # Testing train_model when input ticker value is AAPL
    # Ensure the response is success and indeed returned rmse and mape
    def test_train_model_success(self):
        response = self.client.post(
            reverse('train-model'),
            {'ticker': 'AAPL'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['status'], 'success')
        self.assertIn('rmse', response.json())
        self.assertIn('mape', response.json())

    # Testing train_model when input ticker value is missing
    # Ensure the reponse is 400 bad request with error message
    def test_train_model_missing_ticker(self):
        response = self.client.post(
            reverse('train-model'),
            {},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['error'], 'Ticker symbol is required')

    # Testing get_sentiment_analysis when input ticker value is AAPL
    # Ensure the reponse contains correct data format
    def test_get_sentiment_analysis(self):
        response = self.client.post(
            reverse('get-sentiment-analysis'),
            {'ticker': 'AAPL'},
            content_type='application/json'
        )
        data = response.json()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(data['status'], 'success')
        self.assertIn('average_score', data)
        sentiment_analysis = data['sentiment_analysis']
        self.assertIsInstance(sentiment_analysis, dict)
        expected_columns = {'Sentiment', 'URL', 'Title', 'Description'}
        self.assertTrue(expected_columns.issubset(sentiment_analysis.keys()))
        for key in sentiment_analysis:
            self.assertIsInstance(sentiment_analysis[key], list)

    # Testing get_sentiment_analysis when input ticker value is missing
    # Ensure the reponse is 400 bad request with error message
    def test_get_sentiment_analysis_missing_ticker(self):
        response = self.client.post(
            reverse('get-sentiment-analysis'),
            {},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['error'], 'Ticker symbol is required')
