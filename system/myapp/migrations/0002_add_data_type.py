from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='StockPrediction',
            name='data_type',
            field=models.CharField(
                max_length=10,
                choices=[('train', 'Training'), ('test', 'Testing')],
                default='test'
            ),
        ),
    ]