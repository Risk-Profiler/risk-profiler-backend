#%%
import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000  # Area 1: naik dari 1000 ke 5000

# =========================================================
# RAW FEATURES (tetap messy — untuk latihan data cleaning)
# =========================================================

# 1. ID
merchant_id = [f"UMKM-{str(i).zfill(5)}" for i in range(1, n+1)]

# 2. Kategori bisnis — typo & inkonsistensi huruf
categories = ['F&B', 'fnb', ' F & B ', 'Retail', 'RETAIL',
              'retail ', 'Jasa', 'jasa', 'Fashion', np.nan]
business_category = np.random.choice(categories, n)

# 3. Umur bisnis — ada yang minus (outlier)
business_age_months = np.random.randint(-5, 120, n)

# 4. Volume QRIS — tipe data campuran (ada "Rp", ada integer)
qris_raw = np.random.randint(1000000, 50000000, n)
qris_volume_monthly = [
    f"Rp {val:,.0f}" if np.random.rand() > 0.3 else val
    for val in qris_raw
]

# 5. Hari aktif QRIS — ada yang > 31 (tidak masuk akal)
qris_active_days = np.random.randint(0, 45, n)

# 6. Rating ecommerce — ada 15% missing
ecommerce_rating = np.round(np.random.uniform(1.0, 5.0, n), 1)
ecommerce_rating[np.random.choice(n, int(n * 0.15), replace=False)] = np.nan

# 7. PLN delay — ada missing
pln_delay_days = np.random.choice(
    [0, 2, 5, 15, 30, np.nan], n,
    p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.1]
)

# 8. PDAM
pdam_bill_avg = np.random.randint(50000, 500000, n).astype(float)
pdam_bill_avg[np.random.choice(n, int(n * 0.05), replace=False)] = np.nan

pdam_late_payments = np.random.choice(
    [0, 1, 2, 3, 4, np.nan], n,
    p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]
)

# =========================================================
# BUILD DATAFRAME
# =========================================================

df_messy = pd.DataFrame({
    'merchant_id'         : merchant_id,
    'business_category'   : business_category,
    'business_age_months' : business_age_months,
    'qris_volume_monthly' : qris_volume_monthly,
    'qris_active_days'    : qris_active_days,
    'ecommerce_rating'    : ecommerce_rating,
    'pln_delay_days'      : pln_delay_days,
    'pdam_bill_avg'       : pdam_bill_avg,
    'pdam_late_payments'  : pdam_late_payments,
})

# =========================================================
# CLEANING SEMENTARA — hanya untuk keperluan label generation
# Tidak disimpan, tidak mempengaruhi data messy asli
# =========================================================

def clean_for_labeling(df):
    d = df.copy()

    d['business_age_months'] = d['business_age_months'].abs()

    d['qris_volume_monthly'] = (
        d['qris_volume_monthly']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
    )
    d['qris_volume_monthly'] = pd.to_numeric(
        d['qris_volume_monthly'], errors='coerce'
    ).fillna(0)

    d['qris_active_days']   = d['qris_active_days'].clip(0, 31)
    d['ecommerce_rating']   = d['ecommerce_rating'].fillna(d['ecommerce_rating'].median())
    d['pln_delay_days']     = d['pln_delay_days'].fillna(0)
    d['pdam_late_payments'] = d['pdam_late_payments'].fillna(0)
    d['pdam_bill_avg']      = d['pdam_bill_avg'].fillna(0).abs()

    return d

df_clean = clean_for_labeling(df_messy)

# =========================================================
# AREA 2: HYBRID LABEL ASSIGNMENT
#
# Dua lapis:
# Lapis 1 — ANCHOR: kasus ekstrem yang tidak ambigu
#           Label deterministik, tidak ada randomness
#           Mereplikasi kasus yang analis kredit
#           tidak perlu pikir panjang
#
# Lapis 2 — PROBABILISTIK: kasus abu-abu
#           Ada noise yang realistis
#           Mereplikasi ketidakpastian judgment manusia
# =========================================================

def assign_risk(row):

    # =========================================================
    # LAPIS 1 — ANCHOR CASES
    # Kombinasi indikator yang sangat jelas
    # Tidak ada merchant real yang analis kredit akan debat
    # =========================================================

    # --- ANCHOR HIGH RISK ---

    # Semua indikator pembayaran buruk sekaligus
    if (row['pln_delay_days'] > 14 and
        row['ecommerce_rating'] < 3.0 and
        row['pdam_late_payments'] > 3):
        return 2

    # PLN sangat parah + bisnis sangat baru
    # Bisnis baru yang sudah telat bayar utilitas = red flag besar
    if (row['pln_delay_days'] > 14 and
        row['business_age_months'] < 6):
        return 2

    # PLN parah + PDAM parah + tidak aktif QRIS
    # Tidak ada aktivitas bisnis + semua utilitas telat
    if (row['pln_delay_days'] > 14 and
        row['pdam_late_payments'] > 3 and
        row['qris_active_days'] < 3):
        return 2

    # --- ANCHOR LOW RISK ---

    # Semua indikator prima
    # Merchant ideal — tidak ada celah
    if (row['pln_delay_days'] == 0 and
        row['ecommerce_rating'] >= 4.5 and
        row['pdam_late_payments'] == 0 and
        row['qris_active_days'] >= 20):
        return 0

    # Bisnis sangat mature + tidak pernah telat
    # Track record panjang = established business
    if (row['business_age_months'] > 60 and
        row['pln_delay_days'] == 0 and
        row['pdam_late_payments'] == 0 and
        row['qris_active_days'] >= 25):
        return 0

    # Aktif sangat tinggi + tidak ada delay sama sekali
    # Merchant yang konsisten dan disiplin
    if (row['qris_active_days'] >= 28 and
        row['pln_delay_days'] == 0 and
        row['ecommerce_rating'] >= 4.0 and
        row['pdam_late_payments'] == 0):
        return 0

    # =========================================================
    # LAPIS 2 — PROBABILISTIK
    # Kasus yang genuinely ambigu
    # Satu indikator buruk tapi sisanya oke, atau sebaliknya
    # =========================================================

    score = 0

    # PLN — indikator paling berat
    if row['pln_delay_days'] > 14:
        score += 3
    elif row['pln_delay_days'] > 5:
        score += 1

    # Ecommerce rating
    if row['ecommerce_rating'] < 3.0:
        score += 2
    elif row['ecommerce_rating'] < 4.0:
        score += 1

    # PDAM
    if row['pdam_late_payments'] > 3:
        score += 2
    elif row['pdam_late_payments'] > 1:
        score += 1

    # Umur bisnis
    if row['business_age_months'] < 12:
        score += 1

    # Aktivitas QRIS
    if row['qris_active_days'] < 5:
        score += 1

    # Volume QRIS
    if row['qris_volume_monthly'] < 5_000_000:
        score += 1

    # -------------------------------------------------------
    # Probabilitas lebih tight dari versi sebelumnya
    # Anchor sudah handle kasus ekstrem
    # Sisa hanya kasus yang memang ambigu
    # -------------------------------------------------------

    if score >= 7:
        probs = [0.02, 0.08, 0.90]   # hampir pasti High Risk
    elif score >= 5:
        probs = [0.05, 0.20, 0.75]   # sangat cenderung High Risk
    elif score >= 3:
        probs = [0.20, 0.60, 0.20]   # genuinely Medium Risk
    elif score >= 1:
        probs = [0.75, 0.20, 0.05]   # cenderung Low Risk
    else:
        probs = [0.93, 0.06, 0.01]   # hampir pasti Low Risk

    return np.random.choice([0, 1, 2], p=probs)

# Apply ke semua baris
df_messy['risk_level'] = df_clean.apply(assign_risk, axis=1)

# =========================================================
# VALIDASI DISTRIBUSI
# =========================================================

print("=== RISK LEVEL DISTRIBUTION ===")
print(df_messy['risk_level'].value_counts().sort_index())
print()

dist = df_messy['risk_level'].value_counts(normalize=True)
print("=== DISTRIBUSI CHECK ===")
print(f"Low Risk    (0): {dist.get(0, 0):.1%}  — target ~55-65%")
print(f"Medium Risk (1): {dist.get(1, 0):.1%}  — target ~20-30%")
print(f"High Risk   (2): {dist.get(2, 0):.1%}  — target ~10-20%")

# Cek berapa yang kena anchor vs probabilistik
anchor_high = (
    ((df_clean['pln_delay_days'] > 14) & 
     (df_clean['ecommerce_rating'] < 3.0) & 
     (df_clean['pdam_late_payments'] > 3)) |
    ((df_clean['pln_delay_days'] > 14) & 
     (df_clean['business_age_months'] < 6)) |
    ((df_clean['pln_delay_days'] > 14) & 
     (df_clean['pdam_late_payments'] > 3) & 
     (df_clean['qris_active_days'] < 3))
).sum()

anchor_low = (
    ((df_clean['pln_delay_days'] == 0) & 
     (df_clean['ecommerce_rating'] >= 4.5) & 
     (df_clean['pdam_late_payments'] == 0) & 
     (df_clean['qris_active_days'] >= 20)) |
    ((df_clean['business_age_months'] > 60) & 
     (df_clean['pln_delay_days'] == 0) & 
     (df_clean['pdam_late_payments'] == 0) & 
     (df_clean['qris_active_days'] >= 25)) |
    ((df_clean['qris_active_days'] >= 28) & 
     (df_clean['pln_delay_days'] == 0) & 
     (df_clean['ecommerce_rating'] >= 4.0) & 
     (df_clean['pdam_late_payments'] == 0))
).sum()

print(f"\n=== ANCHOR CASES ===")
print(f"Anchor High Risk : {anchor_high} ({anchor_high/n:.1%})")
print(f"Anchor Low Risk  : {anchor_low} ({anchor_low/n:.1%})")
print(f"Probabilistik    : {n - anchor_high - anchor_low} ({(n - anchor_high - anchor_low)/n:.1%})")

# =========================================================
# SAVE
# =========================================================

df_messy.to_csv('../data/raw/messy_risk_profiler.csv', index=False)

print("\n=== KOLOM YANG DISIMPAN ===")
print(df_messy.columns.tolist())
print(f"\nTotal rows : {len(df_messy)}")
print("Dataset siap!")

# %%