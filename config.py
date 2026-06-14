# =============================================================================
# config.py
# Konfigurasi Global Proyek Watermarking Berbasis Indeks DCT-STDM
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
WATERMARK_BITS = 64          # Panjang indeks biner (64-bit PN Sequence)
REPETITION_FACTOR = 15        # Setiap bit diulang N kali untuk redundansi
DCT_BLOCK_SIZE = 8           # Ukuran blok DCT (standar 8x8)
SVD_SCALING_FACTOR = 250.0   # Faktor Kekuatan SVD (Alpha). Semakin tinggi makin robust, PSNR turun

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

# --- Nama File Dataset CSV ---
DATASET_VIDEO_PATH = os.path.join(DATA_DIR, "dataset_video.csv")
DATASET_LOGO_PATH = os.path.join(DATA_DIR, "dataset_logo.csv")

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

# --- Nama Kolom Fitur Logo (64-bit biner) ---
LOGO_FEATURE_COLUMNS = [f"bit_{i+1}" for i in range(WATERMARK_BITS)]
