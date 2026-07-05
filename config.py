# =============================================================================
# config.py
# Konfigurasi Global Proyek Watermarking Berbasis Indeks DCT-STDM
# Arsitektur Target Y: HIBRIDA (4-Bit Serangan + 1 Scalar Desimal Logo)
# =============================================================================

import os

# --- Direktori Utama ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
LOGOS_DIR = os.path.join(DATA_DIR, "logos")

# Buat direktori jika belum ada
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, SESSIONS_DIR, OUTPUT_DIR, VIDEOS_DIR, LOGOS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Konfigurasi Watermark ---
WATERMARK_BITS = 32          # Panjang indeks biner ASLI zodiak (32-bit payload)
REPETITION_FACTOR = 55        # Setiap bit diulang N kali untuk redundansi
DCT_BLOCK_SIZE = 8           # Ukuran blok DCT (standar 8x8)
SVD_SCALING_FACTOR = 300  # Faktor Kekuatan SVD (Alpha). Semakin tinggi makin robust, PSNR turun

# --- Konfigurasi Error Correction Code (Reed-Solomon) ---
# ECC memperluas 32-bit zodiak menjadi 64-bit (4 Byte data + 4 Byte paritas)
# Mampu memperbaiki hingga 2 Byte (16 bit) yang hancur total
USE_ECC = False              # Matikan ECC untuk melihat kemampuan murni 32-bit
ECC_SYMBOLS = 4              # Jumlah Byte paritas Reed-Solomon (t = ECC_SYMBOLS/2 = 2 Byte error correction)

# --- Konfigurasi Sub-Band DWT untuk Penyisipan ---
# "LL"   = Frekuensi Rendah (paling robust, PSNR lebih rendah)
# "HL"   = Frekuensi Menengah Horizontal (PSNR tinggi, lemah vs Blur/Resize)
# "LH"   = Frekuensi Menengah Vertikal (alternatif HL)
# "DUAL" = Dual-Band Redundancy: sisipkan SIMULTAN di LL dan HL, ekstraksi Fail-Over
#          Keunggulan: Ketahanan absolut vs hampir semua serangan
#          Konsekuensi: Waktu komputasi ~2x lebih lama
DWT_TARGET_SUBBAND = "LL"  # Target sub-band DWT untuk penyisipan watermark

# Posisi koefisien DCT mid-band untuk penyisipan (koordinat dalam blok 8x8)
DCT_MID_BAND = [
    (1, 2), (2, 1), (3, 0),
    (2, 2), (3, 1), (4, 0),
    (3, 2), (4, 1), (5, 0),
]

# --- Konfigurasi Keamanan (Kata Sandi Default untuk Training) ---
TRAINING_PASSWORD = "ZODIAK_TRAINING_2024"

# --- Definisi Kelas Logo Zodiak (8 Kelas) ---
ZODIAK_LABELS = [
    "Aquarius",
    "Cancer",
    "Gemini",
    "Leo",
    "Libra",
    "Pisces",
    "Sagitarius",
    "Scorpio",
]
NUM_ZODIAK = len(ZODIAK_LABELS)  # 8

# --- Definisi Kelas Serangan (9 Kondisi: 1 Bersih + 8 Serangan) ---
ATTACK_LABELS = [
    "Clean",
    "Gaussian_Noise",
    "JPEG_Compression",
    "Blur",
    "Resize",
    "Darkening",
    "Brightening",
    "Rotate",
    "Cropping",
]
NUM_ATTACKS = len(ATTACK_LABELS)  # 9

# --- Konfigurasi Sampling Frame untuk Training ---
FRAMES_PER_SECOND_SAMPLE = 2   # Ambil N frame per detik saat generate dataset
MAX_FRAMES_PER_VIDEO = 150      # Batas maksimum frame per video agar tidak overload

# --- Konfigurasi Machine Learning ---
ML_TEST_SIZE = 0.2              # Porsi data untuk testing (80% train, 20% test)
ML_RANDOM_STATE = 42            # Seed untuk reprodusibilitas
RF_N_ESTIMATORS = 200           # Jumlah pohon dalam Random Forest
RF_MAX_DEPTH = None             # Kedalaman pohon (None = unlimited)

# --- Nama File Model yang Disimpan ---
MODEL_DETECTOR_PATH = os.path.join(MODELS_DIR, "model_detector.pkl")
MODEL_ATTACK_PATH = os.path.join(MODELS_DIR, "model_attack_specialist.pkl")
MODEL_LOGO_PATH = os.path.join(MODELS_DIR, "model_logo_specialist.pkl")
MODEL_MASTER_PATH = os.path.join(MODELS_DIR, "model_master.pkl")  # Model Baru (Gaya Pak Gelar)

# --- Nama File Dataset CSV ---
DATASET_VIDEO_PATH = os.path.join(DATA_DIR, "dataset_video.csv")
DATASET_LOGO_PATH = os.path.join(DATA_DIR, "dataset_logo.csv")
DATASET_MASTER_PATH = os.path.join(DATA_DIR, "dataset_master.csv")  # Dataset Baru (X0+Xw+Xa+Y)

# --- Nama Kolom Fitur (11 Fitur Statistik Video) ---
VIDEO_FEATURE_COLUMNS = [
    "pixel_mean",
    "pixel_variance",
    "pixel_skewness",
    "pixel_kurtosis",
    "dwt_LL_mean",
    "dwt_LL_variance",
    "svd_S_mean",
    "svd_S_variance",
    "edge_density",
    "glcm_contrast",
    "glcm_energy",
]

# --- Nama Kolom Fitur Logo (32-bit biner asli setelah ECC decode) ---
# Catatan: Meskipun 64-bit disisipkan ke video (32 data + 32 paritas ECC),
# kolom yang disimpan ke dataset adalah 32-bit SETELAH ECC mendekode hasilnya.
LOGO_FEATURE_COLUMNS = [f"bit_{i+1}" for i in range(WATERMARK_BITS)]

# =============================================================================
# KONFIGURASI DATASET MASTER (Arsitektur Hibrida: X0 + Xw + Xa -> 5 Kolom Y)
# =============================================================================
# Fitur gabungan: 11 fitur Original + 11 fitur Stego + 11 fitur Attacked = 33 kolom
MASTER_FEATURE_COLUMNS = (
    [f"x0_{c}" for c in VIDEO_FEATURE_COLUMNS] +   # 11 fitur frame Original
    [f"xw_{c}" for c in VIDEO_FEATURE_COLUMNS] +   # 11 fitur frame Stego (sudah disisip)
    [f"xa_{c}" for c in VIDEO_FEATURE_COLUMNS]    # 11 fitur frame Attacked (sudah diserang)
)

# Target Y = Arsitektur HIBRIDA 5 Kolom:
# - 4 Kolom pertama : Representasi Biner Jenis Serangan (4 bit untuk 9 kelas)
# - 1 Kolom terakhir: Skalar Desimal ID Logo Zodiak (angka 0-7 untuk 8 kelas)
# Keunggulan: Menghilangkan Compounding Error pada Random Forest untuk label Logo
# (Random Forest bekerja optimal dengan Scalar Label, bukan Multi-Bit Label)
MASTER_TARGET_COLUMNS = [
    "y_atk_bit1", "y_atk_bit2", "y_atk_bit3", "y_atk_bit4",  # 4 bit serangan
    "y_logo_scalar",                                            # 1 skalar logo (0-7)
]
# Total kolom dataset = 33 fitur X + 5 target Y = 38 kolom per baris

