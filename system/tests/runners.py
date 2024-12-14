import time
from django.test.runner import DiscoverRunner
from django.db import connections
from django.db.utils import OperationalError

class CustomTestRunner(DiscoverRunner):
    def teardown_databases(self, old_config, **kwargs):
        """
        Override teardown_databases to prevent immediate deletion of the database
        if active connections still exist. Retry with a delay before giving up.
        """
        for alias in old_config:
            connection = alias[0]
            db_name = alias[1]
            retries = 1
            wait_time = 5

            for attempt in range(retries):
                try:
                    # Try to destroy the database
                    connection.creation.destroy_test_db(db_name, verbosity=1, keepdb=False)
                    print(f"Successfully destroyed test database for alias '{alias}'.")
                    break  # Exit the retry loop if successful
                except OperationalError as e:
                    print(f"Attempt {attempt + 1}/{retries}: Test database '{db_name}' still has active connections.")
                    time.sleep(wait_time)  # Wait before retrying
            else:
                # If we exhausted retries
                print(f"Warning: Could not destroy test database '{db_name}' due to active connections. Skipping deletion.")
