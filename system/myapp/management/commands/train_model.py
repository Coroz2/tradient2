from django.core.management.base import BaseCommand
from ml.train import train_model

class Command(BaseCommand):
    help = 'Trains the machine learning model'

    def handle(self, *args, **options):
        train_model()
        self.stdout.write(self.style.SUCCESS('Successfully trained model'))