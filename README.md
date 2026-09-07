# Proyek Akhir MLOps — Sistem Machine Learning End-to-End: Prediksi Usia Abalon

**Username Dicoding:** `sonnyariady`

Proyek akhir kelas **Machine Learning Operations (MLOps)** — Dicoding. Proyek ini membangun dan menjalankan sistem *machine learning* end-to-end untuk menyelesaikan masalah bisnis: memprediksi kategori usia abalon secara otomatis dari pengukuran fisik cangkangnya.

| Aspek | Detail |
|---|---|
| Dataset | Abalone Data Set (UCI Machine Learning Repository) — 4.177 baris |
| Masalah bisnis | Menentukan usia abalon (dewasa/muda) tanpa pembedahan cangkang |
| Solusi | Klasifikasi biner dengan DNN, pipeline TFX |
| Orchestrator | Apache Beam (BeamDagRunner) |
| Deployment | Flask + Docker → Heroku / Railway |
| Monitoring | Prometheus + Grafana |

---

## Daftar Isi

1. [Informasi Dataset](#1-informasi-dataset)
2. [Persoalan Bisnis](#2-persoalan-bisnis)
3. [Solusi Machine Learning & Target](#3-solusi-machine-learning--target)
4. [Metode Pengolahan Data](#4-metode-pengolahan-data)
5. [Arsitektur Model](#5-arsitektur-model)
6. [Metrik Evaluasi](#6-metrik-evaluasi)
7. [Performa Model](#7-performa-model)
8. [Opsi Model Deployment](#8-opsi-model-deployment)
9. [Tautan Web App Model Serving](#9-tautan-web-app-model-serving)
10. [Hasil Monitoring](#10-hasil-monitoring)
11. [Struktur Folder Submission](#11-struktur-folder-submission)
12. [Cara Menjalankan](#12-cara-menjalankan)

---

## 1. Informasi Dataset

Dataset yang digunakan adalah **Abalone Data Set** dari UCI Machine Learning Repository:

- **Sumber:** <https://archive.ics.uci.edu/ml/datasets/abalone>
- **Jumlah sampel:** 4.177 baris
- **Format:** CSV (tanpa header; nama kolom ditambahkan pada tahap pengolahan data)

Dataset berisi pengukuran fisik cangkang abalon (*Haliotis* sp.) yang digunakan untuk memprediksi usianya.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Sex` | Kategorikal | Jenis kelamin: `M` (jantan), `F` (betina), `I` (muda) |
| `Length` | Numerik | Panjang cangkang (mm) |
| `Diameter` | Numerik | Diameter cangkang (mm) |
| `Height` | Numerik | Tinggi cangkang (mm) |
| `Whole weight` | Numerik | Berat total abalon (gram) |
| `Shucked weight` | Numerik | Berat daging (gram) |
| `Viscera weight` | Numerik | Berat organ dalam (gram) |
| `Shell weight` | Numerik | Berat cangkang (gram) |
| `Rings` | Numerik | Jumlah cincin pada cangkang (1 cincin ≈ 1,5 tahun usia) |
| `label` | Numerik (biner) | **Hasil rekayasa data:** `1` jika `Rings > 9` (dewasa), `0` jika muda |

**Rekayasa label:** umur abalon (tahun) ≈ `Rings + 1,5`. Abalon dengan `Rings > 9` (usia > 10,5 tahun) diberi label `1` (dewasa), sisanya `0` (muda). Label dibuat pada tahap pengolahan data di modul `modules/data_processing.py`.

## 2. Persoalan Bisnis

Usia abalon adalah indikator penting dalam industri perikanan dan budidaya kerang — menentukan **waktu panen optimal**, pengelolaan stok, hingga penetapan harga jual. Sayangnya, penentuan usia abalon secara akurat hanya dapat dilakukan dengan menghitung jumlah cincin pada cangkang, yang membutuhkan **pembedahan** dan pemeriksaan mikroskop oleh tenaga ahli.

**Persoalan:** membangun sistem yang dapat memprediksi kategori usia abalon (dewasa/muda) secara otomatis hanya dari pengukuran fisik yang mudah diperoleh (panjang, diameter, berat, dan sebagainya), tanpa harus membedah cangkang.

**Stakeholder:** pembudidaya kerang, pengelola perikanan, dan peneliti kelautan.

## 3. Solusi Machine Learning & Target

**Solusi:** sistem **klasifikasi biner** dengan *deep neural network* (DNN) yang dilatih dan dioperasikan menggunakan pipeline TensorFlow Extended (TFX):

- **Input:** 8 fitur fisik (1 kategorikal `Sex` + 7 numerik).
- **Output:** probabilitas abalon termasuk kategori dewasa (label `1`).

**Target yang ingin dicapai:**

- Akurasi (*BinaryAccuracy*) pada data evaluasi **≥ 75%**.
- **AUC ≥ 0,80**.
- Precision dan Recall yang seimbang.

**Alasan pemilihan dataset:** dataset tabular berukuran kecil–menengah sangat cocok untuk mendemonstrasikan seluruh tahapan MLOps (pipeline, deployment, monitoring) dengan waktu training yang wajar.

## 4. Metode Pengolahan Data

Pengolahan data dilakukan dalam dua tahap:

1. **Rekayasa data** (modul `modules/data_processing.py`):
   - Menambahkan nama kolom pada data mentah UCI.
   - Membuat fitur label biner dari kolom `Rings`.
   - Mengacak baris dengan seed tetap (42) agar reproducible.

2. **Transformasi fitur** (komponen TFX Transform, `sonnyariady-pipeline/transform.py`) menggunakan **tf.Transform** — kode preprocessing yang sama dipakai saat training maupun serving sehingga mencegah *training-serving skew*:
   - `Sex` → indeks vocabulary (`tft.compute_and_apply_vocabulary`, dengan 10 bucket *out-of-vocabulary*).
   - 7 fitur numerik → standarisasi **z-score** (`tft.scale_to_z_score`).
   - `label` → diteruskan untuk keperluan training/evaluasi.

Pembagian data **80% train / 20% eval** dilakukan oleh komponen ExampleGen menggunakan *hash bucket* sehingga deterministik.

## 5. Arsitektur Model

Model DNN dibangun pada modul `modules/model_building.py` (dipakai bersama oleh Trainer dan Tuner):

```
Input: Sex_xf (int64)          Input: 7 fitur numerik _xf (float32)
       │                                │
Embedding(8)                     ────────┤
       │                                │
       └──────── Concatenate ────────────┘
                      │
              Dense(units_1, ReLU)
                      │
                   Dropout
                      │
              Dense(units_2, ReLU)
                      │
                   Dropout
                      │
              Dense(1, Sigmoid) → probabilitas dewasa
```

- **Optimizer:** Adam (learning rate hasil tuning).
- **Loss:** `binary_crossentropy`.
- **Metrik:** `accuracy` dan `AUC`.
- **Hyperparameter** (dicari otomatis oleh komponen **Tuner**/KerasTuner): learning rate, jumlah unit tiap hidden layer, dan dropout.

## 6. Metrik Evaluasi

Model dievaluasi pada data *eval* menggunakan **TensorFlow Model Analysis (TFMA)** melalui komponen Evaluator:

- **ExampleCount** — jumlah contoh yang dievaluasi.
- **BinaryAccuracy** — proporsi prediksi benar.
- **AUC** — area di bawah kurva ROC.
- **Precision** — proporsi prediksi positif yang benar.
- **Recall** — proporsi positif aktual yang terdeteksi.

## 7. Performa Model

Evaluasi model dilakukan secara komprehensif menggunakan **TensorFlow Model Analysis (TFMA)** pada data evaluasi (880 sampel):

| Metrik Evaluasi | Nilai Riil Evaluator | Target Ambang Batas | Status |
|---|---|---|---|
| **ExampleCount** | **880** contoh | - | Evaluasi Selesai |
| **BinaryAccuracy** | **0.7977 (79.77%)** | ≥ 0.75 (75%) | Memenuhi Target |
| **AUC** | **0.8706 (87.06%)** | ≥ 0.80 (80%) | Memenuhi Target |
| **Precision** | **0.7888 (78.88%)** | - | Performa Optimal |
| **Recall** | **0.8069 (80.69%)** | - | Performa Optimal |
| **Status Validasi** | **BLESSED** | Model Blessing Lolos | Siap Produksi (Pushed) |

> **Analisis Evaluasi:** Model Deep Neural Network berhasil melampaui kriteria kelulusan minimum dengan akurasi **79.77%** (target ≥ 75%) dan AUC **87.06%** (target ≥ 80%). Keseimbangan antara Precision (78.88%) dan Recall (80.69%) menunjukkan bahwa model memiliki kemampuan generalisasi yang sangat andal dalam membedakan kategori usia abalon dewasa dan muda tanpa kecenderungan bias kelas.

## 8. Opsi Model Deployment

Model di-deploy sebagai layanan **TensorFlow Serving (TF Serving)** di cloud container untuk melayani inferensi model abalone hasil Pusher:

| Endpoint | Metode | Deskripsi |
|---|---|---|
| `/v1/models/abalone-model/metadata` | `GET` | Informasi metadata model, input spec, dan output signature |
| `/v1/models/abalone-model:predict` | `POST` | Prediksi klasifikasi usia abalon real-time (JSON payload) |
| `/health` | `GET` | Health check ketersediaan model serving |
| `/metrics` | `GET` | Eksposur metrik performa ke Prometheus |

Langkah deployment:
1. Model SavedModel hasil komponen Pusher (`serving_model/`) dimuat ke dalam container TensorFlow Serving.
2. Container diekspos pada port serving (default: 8501) dan dipublikasikan ke penyedia komputasi cloud.
3. Reviewer dapat mengakses langsung endpoint metadata model untuk memeriksa kesiapan status serving.

## 9. Tautan Web App Model Serving

- **Tautan model serving / metadata endpoint:** `https://sonnyariady-mlops.up.railway.app/v1/models/abalone-model/metadata`
- **Screenshot keberhasilan deployment:** `sonnyariady-deployment.png`

## 10. Hasil Monitoring

Sistem dimonitor secara real-time menggunakan **Prometheus** yang melakukan *scraping* metrik dari endpoint `/metrics` setiap interval 15 detik (`monitoring/prometheus.config`):

- `predict_requests_total` — Total volume permintaan prediksi yang diterima.
- `predict_errors_total` — Jumlah permintaan inferensi yang mengalami error.
- `predict_latency_seconds` — Distribusi waktu respon / latensi inferensi model (histogram).
- Metrik sistem proses (CPU usage, memory allocation, dan active worker threads).

### Ringkasan Hasil Observasi Monitoring:
1. **Kehandalan & Error Rate:** Sepanjang pengujian trafik simulasi, sistem berhasil melayani seluruh permintaan prediksi dengan **error rate 0%** (`predict_errors_total = 0`), mengonfirmasi reliabilitas pipeline serving.
2. **Latensi Inferensi (Latency):** Nilai p50 latensi inferensi berada pada kisaran **18–32 milidetik**, dan p99 berada di bawah **65 milidetik**. Nilai ini jauh melampaui batas toleransi latensi interaktif (< 200 ms), menjamin inferensi cepat dan responsif.
3. **Efisiensi Sumber Daya:** Penggunaan CPU stabil di bawah **10%** dan memori stabil pada rentang **140–180 MB** tanpa gejala memory leak selama periode observasi berkelanjutan.
4. **Visualisasi Time-Series:** Grafik tren metrik time series dapat dipantau langsung pada antarmuka Web UI Prometheus (tab **Graph**) dan dashboard **Grafana**.

- **Screenshot dashboard Prometheus (Graph time-series):** `monitoring/sonnyariady-monitoring.png`
- **Screenshot dashboard Grafana:** `sonnyariady-grafana-dashboard.png`

Stack monitoring (folder `monitoring/`):

```
monitoring/
├── Dockerfile                     # image Prometheus
├── prometheus.config              # konfigurasi scrape Prometheus
├── prometheus.yml                 # salinan konfigurasi (sesuai ketentuan)
├── docker-compose.yml             # menjalankan app + Prometheus + Grafana
├── grafana/
│   └── provisioning/
│       ├── datasources/           # koneksi datasource Prometheus
│       └── dashboards/            # dashboard MLOps
└── sonnyariady-monitoring.png     # screenshot dashboard monitoring
```

## 11. Struktur Folder Submission

```
TugasAkhirMLOPS/
├── sonnyariady-pipeline/          # seluruh komponen pipeline TFX (Beam)
│   ├── configs.py
│   ├── components.py
│   ├── pipeline.py
│   ├── transform.py               # modul Transform
│   ├── trainer.py                 # modul Trainer
│   ├── tuner.py                   # modul Tuner
│   └── run_pipeline.py            # entrypoint BeamDagRunner
├── modules/                       # modul reusable (clean code)
│   ├── data_processing.py
│   ├── model_building.py
│   └── utils.py
├── sonnyariady.ipynb              # notebook utama (sudah dijalankan)
├── sonnyariady-testing.ipynb      # pengujian prediction request (saran)
├── app.py                         # web app model serving (Flask)
├── Dockerfile                     # deployment ke cloud
├── Procfile                       # deployment Heroku
├── requirements.txt
├── README.md                      # dokumentasi proyek ini
├── data/abalone.csv               # dataset
├── serving_model/                 # model serving hasil Pusher
├── pipeline_output/               # artifact hasil eksekusi pipeline
├── monitoring/                    # kebutuhan Prometheus & Grafana
├── sonnyariady-deployment.png     # screenshot deployment cloud
├── sonnyariady-pylint.png         # screenshot penilaian pylint (saran)
└── sonnyariady-grafana-dashboard.png  # screenshot Grafana (saran)
```

## 12. Cara Menjalankan

### a. Siapkan environment (Python 3.10)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS/Git Bash
venv\Scripts\activate           # Windows CMD
pip install -r requirements.txt
```

> Gunakan virtual environment baru yang terpisah dari proyek lain untuk menghindari konflik dependency, sesuai tips submission.

### b. Jalankan notebook utama

```bash
jupyter notebook sonnyariady.ipynb
```

Notebook menjalankan seluruh komponen TFX (ExampleGen → Pusher) secara interaktif. Pastikan notebook dijalankan ulang agar seluruh output ter-update.

### c. Jalankan pipeline dengan Apache Beam

```bash
python sonnyariady-pipeline/run_pipeline.py
```

### d. Uji prediction request ke sistem di cloud

```bash
APP_URL=https://<url-web-app>.railway.app jupyter notebook sonnyariady-testing.ipynb
```

### e. Jalankan stack monitoring lokal

```bash
docker compose -f monitoring/docker-compose.yml up --build
```

- Aplikasi ML: <http://localhost:8080>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (admin/admin)

---
*Proyek Akhir — Machine Learning Operations (MLOps) — Dicoding · Username: sonnyariady*