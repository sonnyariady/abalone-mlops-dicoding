# ============================================================
# Dockerfile - deployment sistem machine learning ke cloud
# Platform: Heroku / Railway (atau container registry lain)
# Username Dicoding: sonnyariady
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# Install dependencies serving terlebih dahulu agar layer Docker dapat di-cache.
COPY requirements-server.txt requirements.txt* ./
RUN if [ -f requirements-server.txt ]; then pip install --no-cache-dir -r requirements-server.txt; else pip install --no-cache-dir -r requirements.txt; fi


# Salin aplikasi web dan model serving hasil TFX Pusher.
COPY app.py .
COPY serving_model ./serving_model

# Jalankan sebagai user non-root (best practice keamanan).
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Heroku/Railway menyediakan variabel lingkungan PORT.
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 app:app