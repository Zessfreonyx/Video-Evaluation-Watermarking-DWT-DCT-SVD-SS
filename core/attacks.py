# =============================================================================
# core/attacks.py
# Modul Serangan: 8 Fungsi Simulasi Serangan Manipulasi Video
# =============================================================================

import cv2
import numpy as np
import io
from typing import Tuple


def attack_gaussian_noise(frame: np.ndarray, mean: float = 0, std: float = 15) -> np.ndarray:
    """
    Menambahkan Gaussian Noise (derau acak) ke seluruh frame.
    Mensimulasikan gangguan sinyal elektronik pada video.

    Args:
        frame: Frame input (BGR).
        mean: Rata-rata distribusi noise.
        std: Standar deviasi noise (semakin besar = semakin kasar).

    Returns:
        Frame dengan noise Gaussian.
    """
    noise = np.random.normal(mean, std, frame.shape).astype(np.float64)
    noisy = frame.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def attack_jpeg_compression(frame: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Mensimulasikan kompresi JPEG dengan menurunkan kualitasnya.
    Mensimulasikan video yang di-screenshot atau di-upload ke media sosial.

    Args:
        frame: Frame input (BGR).
        quality: Kualitas JPEG (0-100, semakin rendah = semakin rusak).

    Returns:
        Frame setelah dikompres dan di-decompress menggunakan JPEG.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", frame, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def attack_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Menerapkan Gaussian Blur (pengaburan) pada frame.
    Mensimulasikan video yang di-filter atau di-soften.

    Args:
        frame: Frame input (BGR).
        kernel_size: Ukuran kernel (harus angka ganjil, semakin besar = semakin kabur).

    Returns:
        Frame yang sudah dikaburkan.
    """
    # Pastikan kernel_size ganjil
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def attack_resize(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Mengecilkan lalu memperbesar frame kembali ke ukuran asli.
    Mensimulasikan video yang diubah resolusinya (downscale + upscale).

    Args:
        frame: Frame input (BGR).
        scale: Faktor pengecilan (0.5 = dikecilkan 50%, lalu diperbesar kembali).

    Returns:
        Frame yang sudah di-resize bolak-balik.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def attack_darkening(frame: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """
    Menurunkan kecerahan frame secara keseluruhan.
    Mensimulasikan video yang digelapkan oleh editor.

    Args:
        frame: Frame input (BGR).
        factor: Faktor penggelapan (0.0 = hitam total, 1.0 = tidak berubah).

    Returns:
        Frame yang sudah digelapkan.
    """
    darkened = (frame.astype(np.float64) * factor)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def attack_brightening(frame: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Menaikkan kecerahan frame secara keseluruhan.
    Mensimulasikan video yang terlalu terang atau di-overexpose.

    Args:
        frame: Frame input (BGR).
        factor: Faktor pencerahan (> 1.0 = semakin terang).

    Returns:
        Frame yang sudah dicerahkan.
    """
    brightened = (frame.astype(np.float64) * factor)
    return np.clip(brightened, 0, 255).astype(np.uint8)


def attack_rotate(frame: np.ndarray, angle: float = 5.0) -> np.ndarray:
    """
    Memutar frame sebesar N derajat (rotasi kecil).
    Mensimulasikan video yang di-crop atau sedikit diputar oleh editor.

    Args:
        frame: Frame input (BGR).
        angle: Sudut rotasi dalam derajat (positif = berlawanan jarum jam).

    Returns:
        Frame yang sudah diputar (dengan area kosong berwarna hitam).
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def attack_cropping(frame: np.ndarray, crop_ratio: float = 0.1) -> np.ndarray:
    """
    Memotong tepi frame lalu mengembalikannya ke ukuran semula.
    Mensimulasikan cropping manual oleh editor atau konversi format yang mengubah aspect ratio.

    Args:
        frame: Frame input (BGR).
        crop_ratio: Rasio piksel yang dipotong dari setiap sisi (0.1 = 10% dari setiap sisi).

    Returns:
        Frame yang sudah di-crop dan di-resize ke ukuran asli.
    """
    h, w = frame.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    cropped = frame[crop_h:h - crop_h, crop_w:w - crop_w]
    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


# Peta fungsi serangan (nama kelas -> fungsi)
ATTACK_FUNCTIONS = {
    "Clean": lambda f: f.copy(),
    "Gaussian_Noise": attack_gaussian_noise,
    "JPEG_Compression": attack_jpeg_compression,
    "Blur": attack_blur,
    "Resize": attack_resize,
    "Darkening": attack_darkening,
    "Brightening": attack_brightening,
    "Rotate": attack_rotate,
    "Cropping": attack_cropping,
}


def apply_attack(frame: np.ndarray, attack_name: str) -> np.ndarray:
    """
    Menerapkan serangan berdasarkan nama kelas serangan.

    Args:
        frame: Frame input.
        attack_name: Nama serangan (harus ada di ATTACK_LABELS di config.py).

    Returns:
        Frame yang sudah diserang.
    """
    if attack_name not in ATTACK_FUNCTIONS:
        raise ValueError(f"Serangan '{attack_name}' tidak dikenali. Pilihan: {list(ATTACK_FUNCTIONS.keys())}")
    return ATTACK_FUNCTIONS[attack_name](frame)


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: core/attacks.py")
    print("=" * 60)

    dummy_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    for name, func in ATTACK_FUNCTIONS.items():
        result = func(dummy_frame)
        print(f"    [{name:20s}] -> Shape: {result.shape}, dtype: {result.dtype}")

    print("\n[OK] attacks.py berjalan dengan sempurna!")
