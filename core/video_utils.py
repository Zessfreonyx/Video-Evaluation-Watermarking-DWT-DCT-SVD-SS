# =============================================================================
# core/video_utils.py
# Utilitas Video: Baca Frame, Ekstrak 11 Fitur Statistik/Tekstur, Simpan Video
# =============================================================================

import cv2
import numpy as np
from scipy import stats
from scipy.fftpack import dct
from skimage.feature import graycomatrix, graycoprops
from typing import List, Dict, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DCT_MID_BAND,
    DCT_BLOCK_SIZE,
    FRAMES_PER_SECOND_SAMPLE,
    MAX_FRAMES_PER_VIDEO,
    VIDEO_FEATURE_COLUMNS,
)


def read_video_frames(
    video_path: str,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    sample_fps: int = FRAMES_PER_SECOND_SAMPLE,
) -> Tuple[List[np.ndarray], float, int, int]:
    """
    Membaca frame dari video dengan teknik sampling per detik.

    Args:
        video_path: Path menuju file video.
        max_frames: Batas maksimum jumlah frame yang dibaca.
        sample_fps: Jumlah frame yang diambil per detik.

    Returns:
        Tuple: (list frame BGR, fps asli video, lebar, tinggi)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Hitung interval sampling
    if original_fps > 0 and sample_fps > 0:
        interval = max(1, int(original_fps / sample_fps))
    else:
        interval = 10

    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames, original_fps, width, height


def read_all_frames(video_path: str) -> Tuple[List[np.ndarray], float, int, int]:
    """
    Membaca SEMUA frame dari video (digunakan saat proses embed/extract video utuh).

    Returns:
        Tuple: (list frame BGR, fps, lebar, tinggi)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Tidak bisa membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, fps, width, height


def save_video(frames: List[np.ndarray], output_path: str, fps: float, width: int, height: int) -> None:
    """
    Menyimpan daftar frame menjadi file video MP4.

    Args:
        frames: List frame BGR.
        output_path: Path tujuan simpan video (berakhiran .mp4).
        fps: Frame rate video.
        width: Lebar frame.
        height: Tinggi frame.
    """
    if not output_path.endswith(".mp4"):
        output_path = output_path.rsplit(".", 1)[0] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def extract_features(frame: np.ndarray) -> Dict[str, float]:
    """
    Mengekstrak 11 fitur statistik dan tekstur dari sebuah frame video.
    Inilah yang akan digunakan sebagai input (parameter) untuk Model 1 dan Model 2.

    Fitur yang diekstrak:
    1.  pixel_mean       - Rata-rata kecerahan piksel
    2.  pixel_variance   - Varians kecerahan piksel
    3.  pixel_skewness   - Kemiringan distribusi kecerahan
    4.  pixel_kurtosis   - Keruncingan distribusi kecerahan
    5.  dwt_LL_mean      - Rata-rata sub-band LL DWT
    6.  dwt_LL_variance  - Varians sub-band LL DWT
    7.  svd_S_mean       - Rata-rata Singular Value
    8.  svd_S_variance   - Varians Singular Value
    9.  edge_density     - Kepadatan tepi gambar
    10. glcm_contrast    - Kontras tekstur GLCM
    11. glcm_energy      - Energi tekstur GLCM

    Args:
        frame: Frame video (BGR atau Grayscale).

    Returns:
        Dictionary berisi 11 fitur.
    """
    # Konversi ke grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = frame.astype(np.float64)

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    # ---- [1-4] Statistik Piksel ----
    flat = gray.flatten()
    pixel_mean = float(np.mean(flat))
    pixel_variance = float(np.var(flat))
    pixel_skewness = float(stats.skew(flat))
    pixel_kurtosis = float(stats.kurtosis(flat))

    # ---- [5-8] DWT & SVD Features ----
    import pywt
    from scipy.fftpack import dct
    
    # DWT Level 1
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, _ = coeffs2
    
    dwt_LL_mean = float(np.mean(LL))
    dwt_LL_variance = float(np.var(LL))
    
    # Terapkan SVD pada seluruh LL atau DCT dari LL
    # Untuk fitur, kita cukup ambil DCT dari LL secara keseluruhan lalu SVD
    # agar menangkap varians SVD global yang dipengaruhi STDM QIM.
    dct_LL = dct(dct(LL.T, norm='ortho').T, norm='ortho')
    _, S, _ = np.linalg.svd(dct_LL, full_matrices=False)
    
    svd_S_mean = float(np.mean(S))
    svd_S_variance = float(np.var(S))

    # ---- [9] Deteksi Tepi (Canny Edge Density) ----
    h, w = gray_uint8.shape
    edges = cv2.Canny(gray_uint8, 100, 200)
    edge_density = float(np.sum(edges > 0) / (h * w))

    # ---- [10-11] Analisis Tekstur GLCM (Gray-Level Co-occurrence Matrix) ----
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    glcm_energy = float(graycoprops(glcm, 'energy')[0, 0])

    features = {
        "pixel_mean": pixel_mean,
        "pixel_variance": pixel_variance,
        "pixel_skewness": pixel_skewness,
        "pixel_kurtosis": pixel_kurtosis,
        "dwt_LL_mean": dwt_LL_mean,
        "dwt_LL_variance": dwt_LL_variance,
        "svd_S_mean": svd_S_mean,
        "svd_S_variance": svd_S_variance,
        "edge_density": edge_density,
        "glcm_contrast": glcm_contrast,
        "glcm_energy": glcm_energy,
    }
    return features


def get_video_info(video_path: str) -> Dict:
    """Mendapatkan informasi dasar sebuah file video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)),
    }
    cap.release()
    return info


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: core/video_utils.py")
    print("=" * 60)

    # Buat frame uji acak
    dummy_frame = np.random.randint(30, 220, (480, 640, 3), dtype=np.uint8)
    print("\n[1] Ekstrak 11 Fitur dari dummy frame (480x640):")
    features = extract_features(dummy_frame)
    for k, v in features.items():
        print(f"    {k:25s}: {v:.6f}")

    print("\n[OK] video_utils.py berjalan dengan sempurna!")
