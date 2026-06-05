# Risk Profiler Backend

Risk Profiler Backend adalah layanan FastAPI untuk menghasilkan profil risiko kredit UMKM berbasis data alternatif. API ini menerima sinyal operasional seperti volume QRIS, hari aktif QRIS, riwayat utilitas, rating e-commerce, usia usaha, dan kategori bisnis, lalu mengembalikan skor risiko, band kredit, probabilitas kelas risiko, faktor penjelas, serta rekomendasi plafon.

Backend ini dikembangkan untuk kebutuhan PIDI DIGDAYA X HACKATHON 2026 sebagai komponen machine learning service yang dapat diintegrasikan dengan aplikasi frontend, dashboard analis, atau sistem mitra pembiayaan.

## Fitur Utama

- Prediksi risiko UMKM melalui endpoint REST `POST /predict`.
- Model machine learning berbasis pipeline Random Forest yang dimuat dari artefak `joblib`.
- Feature engineering untuk data QRIS, PLN, PDAM, rating e-commerce, usia usaha, dan kategori bisnis.
- Skor risiko 0-100, label risiko, confidence, probabilitas kelas, dan band A-E.
- Explainability berbasis SHAP untuk menampilkan faktor model utama, kontribusi, dan breakdown per kelompok faktor.
- Rekomendasi plafon berbasis skor, kapasitas transaksi, faktor risiko, dan profil pembanding jika data kalibrasi tersedia.
- Endpoint `POST /decisions` untuk mencatat keputusan analis secara sementara di memori aplikasi.
- CORS dikonfigurasi untuk frontend lokal dan deployment frontend Vercel.

## Stack Teknologi

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- Pandas dan NumPy
- Scikit-learn
- SHAP
- Joblib
- Railway untuk deployment backend

## Struktur Proyek

```txt
risk-profiler-backend/
|-- api/
|   |-- main.py                 # FastAPI app, CORS, dan route utama
|   |-- schemas.py              # Pydantic request schema
|   |-- ml_service.py           # Orkestrasi prediksi risiko
|   |-- ml_artifacts.py         # Loader model, scaler, feature list, dan SHAP explainer
|   |-- ml_features.py          # Feature engineering input merchant
|   |-- ml_scoring.py           # Kalkulasi skor, confidence, dan probabilitas kelas
|   |-- ml_calibration.py       # Band, percentile, dan data pembanding
|   |-- ml_explainability.py    # SHAP drivers, kontribusi, dan breakdown faktor
|   |-- ml_recommendations.py   # Rekomendasi plafon dan narasi rekomendasi
|   |-- decision_service.py     # Penyimpanan sementara keputusan analis
|-- data/
|   |-- processed/              # Data training dan data kalibrasi
|   |-- raw/                    # Data mentah
|-- models/
|   |-- random_forest_model.joblib
|   |-- feature_scaler.joblib
|   |-- features_list.joblib
|-- notebooks/                  # Pipeline eksperimen data dan training model
|-- dashboard/                  # Script eksplorasi data
|-- requirements.txt
|-- railway.json
|-- howtouse.md
|-- README.md
```

## Prasyarat

Pastikan Python 3.11 sudah tersedia. Backend membutuhkan artefak model berikut:

```txt
models/random_forest_model.joblib
models/feature_scaler.joblib
models/features_list.joblib
```

File berikut digunakan sebagai data pendukung untuk kalibrasi skor, percentile, band, dan peer comparison:

```txt
data/processed/X_train.csv
data/processed/cleaned_risk_profiler.csv
```

Jika data pendukung tidak tersedia, API tetap dapat berjalan dengan fallback internal untuk threshold band dan batas rekomendasi plafon.

## Instalasi Lokal

Masuk ke folder backend:

```bash
cd risk-profiler-backend
```

Buat dan aktifkan virtual environment.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS atau Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan API:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API akan berjalan di:

```txt
http://127.0.0.1:8000
```

Dokumentasi Swagger tersedia di:

```txt
http://127.0.0.1:8000/docs
```

## Endpoint API

### GET `/health`

Memeriksa apakah backend aktif.

Contoh response:

```json
{
  "status": "ok"
}
```

### POST `/predict`

Menghasilkan prediksi risiko untuk satu merchant UMKM.

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

Field input:

| Field | Type | Keterangan |
| --- | --- | --- |
| `merchant_id` | string | ID unik merchant atau pemohon |
| `business_age_months` | integer | Usia usaha dalam bulan |
| `qris_volume_monthly` | number | Volume transaksi QRIS bulanan |
| `qris_active_days` | integer | Jumlah hari aktif QRIS dalam satu bulan |
| `ecommerce_rating` | number | Rating toko atau reputasi digital |
| `pln_delay_days` | integer | Jumlah hari keterlambatan pembayaran PLN |
| `pdam_bill_avg` | number | Rata-rata tagihan PDAM |
| `pdam_late_payments` | integer | Jumlah keterlambatan pembayaran PDAM |
| `business_category` | string | Kategori usaha, disarankan `fnb`, `fashion`, `jasa`, atau `retail` |

Contoh response ringkas:

```json
{
  "status": "success",
  "data": {
    "merchant_id": "UMKM-TEST",
    "prediction": {
      "risk_level": "Low Risk",
      "score": 86,
      "score_percentile": 0.8125,
      "probability": 0.84,
      "class_probabilities": {
        "Low Risk": 0.84,
        "Medium Risk": 0.12,
        "High Risk": 0.04
      },
      "confidence": "Tinggi",
      "band": "A",
      "band_range": "85-100",
      "recommended_limit": 12500000,
      "peer_comparison_used": true
    }
  }
}
```

Response lengkap pada `data.prediction` juga menyertakan:

```txt
explanation
ai_explanation
shap_drivers
contributions
breakdown
data_sources
recommendations
model_features
```

### POST `/decisions`

Mencatat keputusan analis untuk merchant tertentu.

Contoh request:

```json
{
  "merchant_id": "UMKM-TEST",
  "status": "Approved",
  "note": "Pengajuan diterima berdasarkan hasil analisis risiko.",
  "revision_limit": null
}
```

Nilai `status` yang valid:

```txt
Approved
Rejected
Revision Requested
```

Contoh response:

```json
{
  "status": "success",
  "data": {
    "merchant_id": "UMKM-TEST",
    "status": "Approved",
    "display_status": "Pengajuan diterima",
    "description": "Rekomendasi approval telah dicatat untuk proses lanjutan.",
    "note": "Pengajuan diterima berdasarkan hasil analisis risiko.",
    "revision_limit": null,
    "updated_at": "2026-06-04T17:00:00+00:00"
  }
}
```

Catatan: keputusan analis saat ini disimpan di memori proses backend. Data akan hilang saat server restart dan belum terhubung ke database persisten.

## Smoke Test

Setelah dependency terpasang dan artefak model tersedia, jalankan perintah berikut dari folder backend.

Windows PowerShell:

```bash
.venv\Scripts\python.exe -c "from api.schemas import RiskInput; from api.ml_service import predict_risk; data=RiskInput(merchant_id='MID', business_age_months=14, qris_volume_monthly=8000000, qris_active_days=16, ecommerce_rating=3.8, pln_delay_days=7, pdam_bill_avg=180000, pdam_late_payments=1, business_category='retail'); r=predict_risk(data); print(r['score'], r['band'], r['risk_level'], r['recommended_limit'])"
```

Jika berhasil, terminal akan menampilkan skor, band, label risiko, dan plafon rekomendasi.

## Deployment

Konfigurasi Railway tersedia di `railway.json`.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Pastikan semua artefak model dan file dependency tersedia di repository atau environment deployment.

## CORS

Origin yang saat ini diizinkan:

```txt
http://localhost:3000
http://127.0.0.1:3000
https://risk-profiler-frontend.vercel.app
```

Jika frontend berjalan di domain atau port lain, tambahkan origin tersebut pada konfigurasi `CORSMiddleware` di `api/main.py`.

## Troubleshooting

Jika model gagal dimuat, pastikan file `.joblib` yang dibutuhkan tersedia di folder `models/`.

Jika endpoint `/predict` gagal karena versi library, install ulang dependency dengan:

```bash
pip install -r requirements.txt
```

Jika port 8000 sudah digunakan, jalankan backend di port lain:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
```

Jika port backend berubah, pastikan konfigurasi URL API pada frontend juga diperbarui.

## Tim

- Aditya Cakti Chandrasa: Project Manager
- Nabil Muhammad Hilmi: Lead Engineer
- Zahran Muhammad Syahbana Fardiaz: Machine Learning and Data Pipeline
- Muhammad Ghazi Ali Asy'ary: Risk and Actuary

## Lisensi

Proyek ini dikembangkan untuk Digdaya x Hackathon 2026. Penggunaan, distribusi, dan hak kekayaan intelektual mengikuti ketentuan penyelenggara kompetisi dan kesepakatan tim.
