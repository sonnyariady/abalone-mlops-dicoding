# ============================================================
# Dockerfile - menjalankan Prometheus untuk memantau sistem ML
# Proyek Akhir MLOps - sonnyariady
# ============================================================

FROM prom/prometheus:v2.53.2

# Salin konfigurasi Prometheus (scrape endpoint TF Serving).
COPY prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml"]