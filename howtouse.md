# How To Use - Backend Risk Profiler

Dokumen ini menjelaskan cara menjalankan backend FastAPI untuk Risk Profiler.

## 1. Masuk ke Folder Backend

```bash
cd risk-profiler-backend
```

## 2. Buat Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependency

```bash
pip install -r requirements.txt
```

## 4. Pastikan Artifact Model Tersedia

Backend membutuhkan file berikut:

```txt
models/random_forest_model.joblib
models/feature_scaler.joblib
models/features_list.joblib
```

File data kalibrasi bersifat pendukung:

```txt
data/processed/X_train.csv
data/processed/cleaned_risk_profiler.csv
```

Jika data kalibrasi tersedia, backend dapat menghitung band, percentile, dan rekomendasi plafon dengan profil pembanding. Jika tidak tersedia, sistem tetap memakai fallback internal.

## 5. Jalankan API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Backend berjalan di:

```txt
http://127.0.0.1:8000
```

## 6. Cek Health

Buka:

```txt
http://127.0.0.1:8000/health
```

Response yang benar:

```json
{
  "status": "ok"
}
```

## 7. Dokumentasi API

Swagger:

```txt
http://127.0.0.1:8000/docs
```

## 8. Endpoint Utama

### GET `/health`

Dipakai untuk mengecek backend aktif.

### POST `/predict`

Dipakai frontend untuk membuat prediksi risiko UMKM.

Contoh request:

```json
{
  "merchant_id": "UMKM-TEST",
  "business_age_months": 24,
  "qris_volume_monthly": 8000000,
  "qris_active_days": 22,
  "ecommerce_rating": 4.5,
  "pln_delay_days": 2,
  "pdam_bill_avg": 150000,
  "pdam_late_payments": 0,
  "business_category": "fnb"
}
```

Response utama berada di:

```txt
data.prediction
```

Field penting:

```txt
risk_level
score
band
band_range
recommended_limit
peer_comparison_used
recommendations
shap_drivers
breakdown
data_sources
```

### POST `/decisions`

Dipakai untuk mencatat keputusan analis.

Contoh request:

```json
{
  "merchant_id": "UMKM-TEST",
  "status": "Approved",
  "note": "Pengajuan diterima berdasarkan hasil analisis risiko.",
  "revision_limit": null
}
```

Status yang valid:

```txt
Approved
Rejected
Revision Requested
```

Catatan: endpoint decision saat ini menyimpan data sementara di memory backend. Data akan hilang jika server dimatikan.

## 9. Smoke Test Prediksi

Windows PowerShell:

```bash
.venv\Scripts\python.exe -c "from api.schemas import RiskInput; from api.ml_service import predict_risk; data=RiskInput(merchant_id='MID', business_age_months=14, qris_volume_monthly=8000000, qris_active_days=16, ecommerce_rating=3.8, pln_delay_days=7, pdam_bill_avg=180000, pdam_late_payments=1, business_category='retail'); r=predict_risk(data); print(r['score'], r['band'], r['risk_level'], r['recommended_limit'])"
```

Jika berhasil, terminal akan menampilkan skor, band, level risiko, dan plafon rekomendasi.

## 10. Troubleshooting

### Frontend tidak bisa fetch API

Pastikan backend berjalan di port yang sama dengan konfigurasi frontend:

```txt
http://127.0.0.1:8000
```

### CORS error

Backend saat ini mengizinkan origin:

```txt
http://localhost:3000
http://127.0.0.1:3000
```

Jika frontend dijalankan di port lain, tambahkan origin tersebut di `api/main.py`.

### Model gagal load

Pastikan file `.joblib` tersedia di folder `models`.

### SHAP atau scikit-learn error

Install ulang dependency:

```bash
pip install -r requirements.txt
```

### Port 8000 sudah dipakai

Gunakan port lain:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
```

Jika port diganti, update juga `NEXT_PUBLIC_RISK_API_URL` di frontend.
