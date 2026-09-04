# Procfile untuk deployment di Heroku (alternatif: Railway dengan Dockerfile).
web: gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app