import os
import zipfile

OUTPUT_ZIP = "submission_sonnyariady.zip"
INCLUDE_FILES = [
    "sonnyariady.ipynb",
    "sonnyariady-testing.ipynb",
    "local_pipeline.py",
    "app.py",
    "Dockerfile",
    "Procfile",
    "requirements.txt",
    "requirements-server.txt",
    "README.md",
    "sonnyariady-deployment.png",
    "sonnyariady-monitoring.png",
    "sonnyariady-grafana-dashboard.png",
    "sonnyariady-pylint.png",
    "monitoring.png",
]
INCLUDE_DIRS = [
    "data",
    "modules",
    "sonnyariady-pipeline",
    "serving_model",
    "monitoring",
]
EXCLUDE_EXTS = [".pyc"]
EXCLUDE_DIRS = ["__pycache__", ".ipynb_checkpoints"]

def build_zip():
    print(f"Mengemas submission ke {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            if os.path.exists(f):
                z.write(f, f)
                print(f"  + {f}")
            else:
                print(f"  [WARNING] File tidak ditemukan: {f}")
        for d in INCLUDE_DIRS:
            if os.path.exists(d):
                for root, dirs, files in os.walk(d):
                    dirs[:] = [d_name for d_name in dirs if d_name not in EXCLUDE_DIRS]
                    for file in files:
                        if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        z.write(full_path, rel_path)
    print(f"Selesai! Ukuran {OUTPUT_ZIP}: {os.path.getsize(OUTPUT_ZIP) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    build_zip()
