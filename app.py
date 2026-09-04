"""Web app model serving untuk sistem machine learning prediksi usia abalon.

Dibangun dengan Flask dan melayani SavedModel hasil komponen TFX Pusher.
Seluruh metrik proses dipublikasikan pada endpoint ``/metrics`` dan
dikumpulkan oleh Prometheus.

Endpoint:
    GET  /         : informasi layanan.
    GET  /health   : health check.
    POST /predict  : prediksi usia abalon dari fitur mentah (JSON).
    GET  /metrics  : metrik Prometheus (text format).

Cara menjalankan lokal:
    python app.py            (development server, port 8080)
    gunicorn app:app         (production server)
"""

import os
import time

import tensorflow as tf
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    REGISTRY,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "serving_model")
PORT = int(os.environ.get("PORT", "8080"))

# Fitur mentah yang wajib dikirim klien (nama kolom dataset abalone).
REQUIRED_FEATURES = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]

# ---------------------------------------------------------------------------
# Metrik Prometheus
# ---------------------------------------------------------------------------
PREDICT_REQUESTS = Counter(
    "predict_requests_total", "Total permintaan prediksi yang diterima"
)
PREDICT_ERRORS = Counter(
    "predict_errors_total", "Total permintaan prediksi yang gagal"
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latensi proses prediksi (detik)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _load_model() -> tf.Module:
    """Memuat SavedModel hasil TFX Pusher beserta signature serving-nya."""
    model_path = MODEL_DIR
    if os.path.exists(model_path):
        subdirs = [
            os.path.join(model_path, d)
            for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        if subdirs and not os.path.exists(os.path.join(model_path, "saved_model.pb")):
            model_path = max(subdirs, key=os.path.getmtime)
    return tf.saved_model.load(model_path)


model = _load_model()


def _build_tf_example(features: dict) -> tf.train.Example:
    """Membangun ``tf.train.Example`` dari dict fitur mentah.

    Fitur ``Sex`` diperlakukan sebagai string, sisanya sebagai float.
    """
    feature = {}
    for key, value in features.items():
        if key == "Sex":
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(value).encode("utf-8")])
            )
        else:
            feature[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(value)])
            )
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Informasi dasar layanan."""
    return jsonify(
        {
            "service": "abalone-age-prediction",
            "description": (
                "Model serving untuk memprediksi usia abalon "
                "(klasifikasi biner: dewasa/muda)."
            ),
            "endpoints": ["/health", "/predict", "/metrics"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check sederhana."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """Menerima JSON berisi fitur abalon dan mengembalikan probabilitas.

    Format payload:
        {"features": [{"Sex": "M", "Length": 0.455, ...}, ...]}
    atau dict tunggal:
        {"Sex": "M", "Length": 0.455, ...}
    """
    PREDICT_REQUESTS.inc()
    start_time = time.perf_counter()
    try:
        payload = request.get_json(force=True)
        features_list = payload.get("features") or payload.get("instances")
        if isinstance(features_list, dict):
            features_list = [features_list]
        if not features_list:
            return (
                jsonify(
                    {"error": "Field 'features' harus berisi minimal satu contoh."}
                ),
                400,
            )

        # Validasi kelengkapan fitur.
        for example in features_list:
            missing = [key for key in REQUIRED_FEATURES if key not in example]
            if missing:
                return (
                    jsonify(
                        {"error": f"Fitur berikut tidak ditemukan: {missing}"}
                    ),
                    400,
                )

        serialized = [
            _build_tf_example(example).SerializeToString() for example in features_list
        ]
        input_tensor = tf.constant(serialized, dtype=tf.string)

        probabilities = model.signatures["serving_default"](
            examples=input_tensor
        )["outputs"].numpy()

        predictions = [
            {
                "probability": round(float(prob[0]), 4),
                "prediction": int(prob[0] >= 0.5),
                "label": "dewasa" if prob[0] >= 0.5 else "muda",
            }
            for prob in probabilities
        ]
        return jsonify({"predictions": predictions})
    except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
        PREDICT_ERRORS.inc()
        return jsonify({"error": str(exc)}), 500
    finally:
        PREDICT_LATENCY.observe(time.perf_counter() - start_time)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint metrik Prometheus (text format)."""
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
