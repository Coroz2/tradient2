# Generated by Django 4.2.16 on 2024-10-29 02:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="StockPrediction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("ticker", models.CharField(max_length=10)),
                ("date", models.DateTimeField()),
                ("actual_price", models.FloatField()),
                ("predicted_price", models.FloatField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-date"],
            },
        ),
    ]
