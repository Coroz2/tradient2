from django.db import models


class StockPrediction(models.Model):
    ticker = models.CharField(max_length=10)
    date = models.DateTimeField()
    actual_price = models.FloatField()
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.ticker} - {self.date.strftime('%Y-%m-%d')}"
