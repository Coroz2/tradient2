# main-project-team61
# Tradient -- About Us



# Technical Architecture





# Project Set up

### Create a virtual environment
```
python3.11 -m venv .venv
```
### You may need to install that version of python
```
brew install python@3.11
```
### Activate on macOS and Linux:
```
source .venv/bin/activate
```
### Install dependencies
```
pip install -r requirements.txt
```


### Start the frontend
```
cd frontend
npm install
```

### Start the Django server (not needed rn)
```
python manage.py runserver 8080
```


# Run Testing
### Testing URLs Deployment
```
python manage.py test system.tests.test_urls
```
### Testing Database Connection
```
python manage.py test system.tests.test_supabase_client
```
### Testing Models
```
python manage.py test system.tests.test_views
```


# Our Roles
