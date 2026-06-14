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
