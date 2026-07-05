# =============================================================================
# core/security.py
# Modul Keamanan: Generator Kunci Temporal, Kunci Spasial, dan Indeks Zodiak
# =============================================================================

import hashlib
import numpy as np
from typing import List


def _password_to_seed(password: str) -> int:
    """Mengubah string password menjadi bilangan bulat seed menggunakan SHA-256."""
    hash_bytes = hashlib.sha256(password.encode("utf-8")).digest()
    # Ambil 8 byte pertama sebagai integer (64-bit)
    seed = int.from_bytes(hash_bytes[:8], byteorder="big")
    # Batasi agar sesuai range numpy seed (0 sampai 2^32 - 1)
    return seed % (2**32)


def generate_temporal_key(password: str, total_frames: int, num_frames_to_embed: int) -> List[int]:
    """
    Membuat daftar indeks frame yang dipilih secara acak berdasarkan password.
    Frame-frame inilah yang akan disisipi watermark.

    Args:
        password: Kata sandi rahasia pengguna.
        total_frames: Jumlah total frame dalam video.
        num_frames_to_embed: Jumlah frame yang ingin disisipkan watermark.

    Returns:
        List of sorted frame indices.
    """
    seed = _password_to_seed(password + "_temporal")
    rng = np.random.RandomState(seed)

    num_select = min(num_frames_to_embed, total_frames)
    selected_indices = sorted(
        rng.choice(total_frames, size=num_select, replace=False).tolist()
    )
    return selected_indices


def generate_spatial_key(password: str, length: int) -> np.ndarray:
    """
    Membuat vektor spread (PN Sequence) untuk digunakan sebagai Spatial Key.
    Vektor ini menentukan arah tebaran bit di dalam domain DCT.

    Args:
        password: Kata sandi rahasia pengguna.
        length: Panjang vektor yang diinginkan.

    Returns:
        numpy array berisi nilai +1 atau -1 (bipolar PN Sequence).
    """
    seed = _password_to_seed(password + "_spatial")
    rng = np.random.RandomState(seed)
    # Generate vektor bipolar: 0 menjadi -1, 1 menjadi +1
    raw = rng.randint(0, 2, size=length)
    spread_vector = np.where(raw == 0, -1, 1).astype(np.float64)
    return spread_vector


def generate_zodiak_index(zodiak_name: str, num_bits: int = 64) -> np.ndarray:
    """
    Membuat indeks biner unik (PN Sequence) untuk setiap nama zodiak.
    Menggunakan hashing dari nama zodiak sebagai seed agar hasilnya
    selalu deterministik (nama yang sama selalu menghasilkan bit yang sama).

    Args:
        zodiak_name: Nama zodiak (misal: "Aries", "Taurus").
        num_bits: Panjang indeks biner yang dihasilkan (default: 64 bit).

    Returns:
        numpy array berisi nilai 0 atau 1 (64-bit PN Sequence).
    """
    # Gunakan nama zodiak dalam huruf besar + salt unik sebagai seed
    seed = _password_to_seed(zodiak_name.upper() + "_ZODIAK_INDEX")
    rng = np.random.RandomState(seed)
    bit_sequence = rng.randint(0, 2, size=num_bits).astype(np.uint8)
    return bit_sequence


def verify_password_strength(password: str) -> bool:
    """
    Memeriksa apakah password cukup kuat (minimal 6 karakter).

    Args:
        password: Kata sandi yang akan diperiksa.

    Returns:
        True jika password valid, False jika tidak.
    """
    return len(password) >= 6


# =============================================================================
# SKEMA PAK GELAR: Konversi Label ke Biner & Encoding Target Y
# =============================================================================

def zodiak_to_3bit(zodiak_name: str) -> np.ndarray:
    """
    Mengonversi nama zodiak menjadi representasi 3-bit biner.
    log2(8 kelas) = 3 bit. Mapping berdasarkan urutan alfabetis ZODIAK_LABELS.

    Args:
        zodiak_name: Nama zodiak (misal: "Leo").

    Returns:
        numpy array berisi 3 bit biner (misal: [0, 1, 1] untuk Leo).
    """
    from config import ZODIAK_LABELS
    if zodiak_name not in ZODIAK_LABELS:
        raise ValueError(f"Zodiak tidak dikenal: {zodiak_name}. Pilihan: {ZODIAK_LABELS}")
    idx = ZODIAK_LABELS.index(zodiak_name)
    # Konversi index (0-7) ke 3-bit binary
    bits = [(idx >> (2 - i)) & 1 for i in range(3)]
    return np.array(bits, dtype=np.uint8)


def attack_to_4bit(attack_name: str) -> np.ndarray:
    """
    Mengonversi nama serangan menjadi representasi 4-bit biner.
    ceil(log2(9 kelas)) = 4 bit. Mapping berdasarkan urutan ATTACK_LABELS.

    Args:
        attack_name: Nama serangan (misal: "Blur").

    Returns:
        numpy array berisi 4 bit biner (misal: [0, 0, 1, 1] untuk Blur).
    """
    from config import ATTACK_LABELS
    if attack_name not in ATTACK_LABELS:
        raise ValueError(f"Serangan tidak dikenal: {attack_name}. Pilihan: {ATTACK_LABELS}")
    idx = ATTACK_LABELS.index(attack_name)
    # Konversi index (0-8) ke 4-bit binary
    bits = [(idx >> (3 - i)) & 1 for i in range(4)]
    return np.array(bits, dtype=np.uint8)


def encode_labels_to_7bit(attack_name: str, zodiak_name: str) -> np.ndarray:
    """
    Mengonversi kombinasi nama serangan + nama zodiak menjadi array 7 bit biner.
    Format: [4 bit serangan | 3 bit logo]
    Ini adalah format Target Y yang digunakan untuk Multi-Output Classifier.

    Args:
        attack_name: Nama serangan (misal: "Blur").
        zodiak_name: Nama zodiak (misal: "Leo").

    Returns:
        numpy array berisi 7 bit biner, misal: [0, 0, 1, 1, 0, 1, 1] untuk Blur+Leo.
    """
    atk_bits = attack_to_4bit(attack_name)    # 4 bit
    logo_bits = zodiak_to_3bit(zodiak_name)   # 3 bit
    return np.concatenate([atk_bits, logo_bits])  # gabung jadi 7 bit


def decode_7bit_to_labels(bits_7: np.ndarray) -> tuple:
    """
    Mendekode array 7 bit biner kembali menjadi nama serangan dan nama zodiak.
    Format input: [4 bit serangan | 3 bit logo]

    Args:
        bits_7: numpy array atau list berisi 7 angka biner (0 atau 1).

    Returns:
        Tuple (attack_name: str, zodiak_name: str).
    """
    from config import ZODIAK_LABELS, ATTACK_LABELS
    bits_7 = [int(b) for b in bits_7]

    # Decode 4 bit pertama -> indeks serangan
    attack_idx = (bits_7[0] << 3) | (bits_7[1] << 2) | (bits_7[2] << 1) | bits_7[3]
    # Decode 3 bit terakhir -> indeks zodiak
    zodiak_idx = (bits_7[4] << 2) | (bits_7[5] << 1) | bits_7[6]

    # Clamp agar tidak keluar batas (safety guard)
    attack_idx = min(attack_idx, len(ATTACK_LABELS) - 1)
    zodiak_idx = min(zodiak_idx, len(ZODIAK_LABELS) - 1)

    return ATTACK_LABELS[attack_idx], ZODIAK_LABELS[zodiak_idx]


# =============================================================================
# SKEMA HIBRIDA: 4-Bit Serangan + 1 Skalar Desimal Logo
# Keunggulan: Menghilangkan Compounding Error pada Random Forest untuk Logo
# =============================================================================

def encode_hybrid_labels(attack_name: str, zodiak_name: str) -> np.ndarray:
    """
    Mengonversi kombinasi nama serangan + nama zodiak menjadi array 5 nilai.
    Format HIBRIDA: [4 bit serangan (biner) | 1 skalar logo (desimal 0-7)]

    Keunggulan dibanding 7-bit murni: Random Forest bekerja optimal pada
    Scalar Label (1 kolom angka utuh) untuk klasifikasi multi-kelas logo,
    menghindari Ordinal Fallacy dan Compounding Error.

    Args:
        attack_name: Nama serangan (misal: \"Blur\").
        zodiak_name: Nama zodiak (misal: \"Leo\").

    Returns:
        numpy array berisi 5 nilai: [bit1, bit2, bit3, bit4, scalar_logo]
        Contoh: Blur + Leo -> [0, 0, 1, 1, 3]
    """
    from config import ZODIAK_LABELS
    atk_bits = attack_to_4bit(attack_name)         # 4 bit biner serangan
    zodiak_scalar = ZODIAK_LABELS.index(zodiak_name)  # 1 angka desimal (0-7)
    return np.append(atk_bits, zodiak_scalar)       # gabung jadi 5 nilai


def decode_hybrid_labels(cols_5) -> tuple:
    """
    Mendekode array 5 nilai (Arsitektur Hibrida) kembali menjadi nama serangan
    dan nama zodiak.
    Format input: [4 bit serangan (biner) | 1 skalar logo (desimal 0-7)]

    Args:
        cols_5: array/list berisi 5 angka. 4 pertama biner, 1 terakhir skalar.

    Returns:
        Tuple (attack_name: str, zodiak_name: str).
    """
    from config import ZODIAK_LABELS, ATTACK_LABELS
    cols_5 = [int(c) for c in cols_5]

    # Decode 4 bit pertama -> indeks serangan
    attack_idx = (cols_5[0] << 3) | (cols_5[1] << 2) | (cols_5[2] << 1) | cols_5[3]
    # Kolom ke-5 adalah langsung indeks zodiak (skalar)
    zodiak_idx = cols_5[4]

    # Clamp agar tidak keluar batas (safety guard)
    attack_idx = min(attack_idx, len(ATTACK_LABELS) - 1)
    zodiak_idx = min(zodiak_idx, len(ZODIAK_LABELS) - 1)

    return ATTACK_LABELS[attack_idx], ZODIAK_LABELS[zodiak_idx]


# =============================================================================
# DEMO: Jalankan file ini langsung untuk melihat output
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: core/security.py")
    print("=" * 60)

    pwd = "RAHASIA123"
    total = 500
    to_embed = 30

    print(f"\n[1] Temporal Key (password='{pwd}', total_frame={total}, embed={to_embed})")
    t_key = generate_temporal_key(pwd, total, to_embed)
    print(f"    Frame yang dipilih: {t_key[:10]}... (total {len(t_key)} frame)")

    print(f"\n[2] Spatial Key (panjang=9)")
    s_key = generate_spatial_key(pwd, 9)
    print(f"    Spread Vector: {s_key}")

    print(f"\n[3] Indeks Zodiak (64-bit PN Sequence)")
    from config import ZODIAK_LABELS
    for z in ZODIAK_LABELS:
        idx = generate_zodiak_index(z)
        print(f"    {z:10s}: {idx[:16]}... (total {len(idx)} bit)")

    print("\n[OK] security.py berjalan dengan sempurna!")
