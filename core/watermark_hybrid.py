# =============================================================================
# core/watermark_hybrid.py
# Implementasi Hybrid DWT-DCT-SVD dengan Spread Spectrum
# Mendukung mode penyisipan: LL, HL, LH, dan DUAL (LL + HL simultan)
# Mode DUAL menggunakan mekanisme Fail-Over ECC untuk ekstraksi paling tangguh
# =============================================================================

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from config import (
    DCT_BLOCK_SIZE, SVD_SCALING_FACTOR, REPETITION_FACTOR,
    WATERMARK_BITS, USE_ECC, ECC_SYMBOLS, DWT_TARGET_SUBBAND
)
from core.security import generate_spatial_key, _password_to_seed

try:
    import reedsolo
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

def _get_shuffled_blocks(password: str, blocks_h: int, blocks_w: int, total_bits: int):
    all_blocks = [(i, j) for i in range(blocks_h) for j in range(blocks_w)]
    seed_int = _password_to_seed(password)
    rng = np.random.RandomState(seed_int)
    rng.shuffle(all_blocks)
    return all_blocks[:total_bits]


def _ecc_encode(bitstream: np.ndarray) -> np.ndarray:
    """
    Meng-enkode bitstream menggunakan Reed-Solomon.
    Input : N bit (harus kelipatan 8, misal 32 bit = 4 Byte)
    Output: N bit + (ECC_SYMBOLS * 8) bit paritas
    Contoh: 32 bit -> 64 bit (4 Byte data + 4 Byte paritas RS)
    """
    if not USE_ECC or not _RS_AVAILABLE:
        return bitstream

    # Ubah bit array menjadi bytes
    n_bytes = len(bitstream) // 8
    byte_list = []
    for i in range(n_bytes):
        byte_val = int("".join(str(b) for b in bitstream[i*8:(i+1)*8]), 2)
        byte_list.append(byte_val)

    # Enkode dengan Reed-Solomon
    rs = reedsolo.RSCodec(ECC_SYMBOLS)
    encoded_bytes = bytes(rs.encode(bytes(byte_list)))

    # Ubah kembali bytes menjadi bit array
    encoded_bits = []
    for byte_val in encoded_bytes:
        bits = [(byte_val >> (7 - i)) & 1 for i in range(8)]
        encoded_bits.extend(bits)

    return np.array(encoded_bits, dtype=np.uint8)


def _ecc_decode(encoded_bits: np.ndarray) -> np.ndarray:
    """
    Mendekode bitstream yang sudah di-enkode Reed-Solomon.
    Input : N bit encoded (misal 64 bit)
    Output: N bit data asli (misal 32 bit)
    Jika error terlalu parah, fallback ke 32 bit pertama mentah.
    """
    if not USE_ECC or not _RS_AVAILABLE:
        return encoded_bits

    # Ubah bit array menjadi bytes
    n_bytes = len(encoded_bits) // 8
    byte_list = []
    for i in range(n_bytes):
        byte_val = int("".join(str(b) for b in encoded_bits[i*8:(i+1)*8].astype(int)), 2)
        byte_list.append(byte_val)

    rs = reedsolo.RSCodec(ECC_SYMBOLS)
    try:
        # Decode dan perbaiki error
        decoded_result = rs.decode(bytes(byte_list))
        # decode() mengembalikan tuple (decoded_msg, decoded_msgecc, errata_pos)
        decoded_bytes = bytes(decoded_result[0])

        # Ubah kembali bytes menjadi bit array
        decoded_bits = []
        for byte_val in decoded_bytes:
            bits = [(byte_val >> (7 - i)) & 1 for i in range(8)]
            decoded_bits.extend(bits)

        return np.array(decoded_bits, dtype=np.uint8)
    except Exception:
        # Fallback: error terlalu parah, kembalikan bit data asli mentah
        # (ambil bagian data, buang paritas)
        data_bit_len = len(encoded_bits) - (ECC_SYMBOLS * 8)
        return encoded_bits[:data_bit_len].astype(np.uint8)


def _embed_to_band(band: np.ndarray, modulated_bits: np.ndarray, password: str, alpha: float) -> np.ndarray:
    """
    Helper: Menyisipkan modulated_bits ke dalam satu sub-band DWT menggunakan DCT-SVD.
    Digunakan oleh embed_bitstream untuk mode tunggal maupun DUAL.
    """
    h, w = band.shape
    blocks_h = h // DCT_BLOCK_SIZE
    blocks_w = w // DCT_BLOCK_SIZE
    total_bits = len(modulated_bits)

    if blocks_h * blocks_w < total_bits:
        raise ValueError(
            f"Sub-band terlalu kecil untuk memuat {total_bits} bit. "
            f"Pastikan resolusi video minimal 480p."
        )

    band_embedded = band.copy()
    selected_blocks = _get_shuffled_blocks(password, blocks_h, blocks_w, total_bits)

    for bit_idx, (i, j) in enumerate(selected_blocks):
        r_start, r_end = i * DCT_BLOCK_SIZE, (i + 1) * DCT_BLOCK_SIZE
        c_start, c_end = j * DCT_BLOCK_SIZE, (j + 1) * DCT_BLOCK_SIZE
        block = band_embedded[r_start:r_end, c_start:c_end]

        dct_block = apply_dct(block)
        U, S, V = np.linalg.svd(dct_block, full_matrices=False)

        m = 0 if modulated_bits[bit_idx] == -1 else 1
        quantized = np.round((S[0] - m * (alpha / 2)) / alpha) * alpha + m * (alpha / 2)
        S[0] = quantized

        dct_reconstructed = np.dot(U, np.dot(np.diag(S), V))
        block_reconstructed = apply_idct(dct_reconstructed)
        band_embedded[r_start:r_end, c_start:c_end] = block_reconstructed

    return band_embedded


def _extract_from_band(band: np.ndarray, password: str, alpha: float, num_encoded_bits: int) -> np.ndarray:
    """
    Helper: Mengekstrak encoded_bits dari satu sub-band DWT menggunakan blind DCT-SVD.
    Mengembalikan array encoded_bits (sebelum ECC decode).
    """
    h, w = band.shape
    blocks_h = h // DCT_BLOCK_SIZE
    blocks_w = w // DCT_BLOCK_SIZE
    total_bits = num_encoded_bits * REPETITION_FACTOR
    spatial_key = generate_spatial_key(password, total_bits)

    extracted_modulated = []
    selected_blocks = _get_shuffled_blocks(password, blocks_h, blocks_w, total_bits)

    for bit_idx, (i, j) in enumerate(selected_blocks):
        r_start, r_end = i * DCT_BLOCK_SIZE, (i + 1) * DCT_BLOCK_SIZE
        c_start, c_end = j * DCT_BLOCK_SIZE, (j + 1) * DCT_BLOCK_SIZE
        block = band[r_start:r_end, c_start:c_end]

        dct_block = apply_dct(block)
        U, S, V = np.linalg.svd(dct_block, full_matrices=False)
        val = S[0]

        dist_0 = abs(val - (np.round(val / alpha) * alpha))
        dist_1 = abs(val - (np.round((val - alpha / 2) / alpha) * alpha + alpha / 2))

        extracted_modulated.append(1 if dist_1 < dist_0 else -1)

    extracted_modulated = np.array(extracted_modulated)
    demodulated = extracted_modulated * spatial_key
    demodulated_reshaped = demodulated.reshape((num_encoded_bits, REPETITION_FACTOR))
    sums = np.sum(demodulated_reshaped, axis=1)
    encoded_bits = np.where(sums > 0, 1, 0).astype(np.uint8)
    return encoded_bits


def _try_ecc_decode_strict(encoded_bits: np.ndarray) -> np.ndarray | None:
    """
    Mencoba ECC decode secara KETAT (strict). Mengembalikan None jika gagal (tidak ada fallback).
    Digunakan oleh Fail-Over Mechanism agar bisa mendeteksi kegagalan secara eksplisit.
    """
    if not USE_ECC or not _RS_AVAILABLE:
        return encoded_bits

    n_bytes = len(encoded_bits) // 8
    byte_list = []
    for i in range(n_bytes):
        byte_val = int("".join(str(b) for b in encoded_bits[i*8:(i+1)*8].astype(int)), 2)
        byte_list.append(byte_val)

    rs = reedsolo.RSCodec(ECC_SYMBOLS)
    try:
        decoded_result = rs.decode(bytes(byte_list))
        decoded_bytes = bytes(decoded_result[0])
        decoded_bits = []
        for byte_val in decoded_bytes:
            bits = [(byte_val >> (7 - i)) & 1 for i in range(8)]
            decoded_bits.extend(bits)
        return np.array(decoded_bits, dtype=np.uint8)
    except Exception:
        # Kembalikan None sebagai sinyal kegagalan (tanpa fallback) untuk Fail-Over
        return None


def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def embed_bitstream(frame: np.ndarray, bitstream: np.ndarray, password: str, alpha: float = SVD_SCALING_FACTOR) -> np.ndarray:
    """
    Menyisipkan watermark ke dalam frame video menggunakan DWT-DCT-SVD-SS + ECC.

    Mode penyisipan dikontrol oleh DWT_TARGET_SUBBAND di config.py:
    - "LL", "HL", "LH" : Penyisipan ke satu sub-band
    - "DUAL"           : Penyisipan SIMULTAN ke LL dan HL (Dual-Band Redundancy)
    """
    # 1. Konversi ke YCrCb dan ambil channel Luminance (Y)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # 2. DWT Level 1
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')

    # 3. ECC Encode (32 bit -> 64 bit dengan 4 Byte paritas RS)
    encoded_bitstream = _ecc_encode(bitstream)
    num_encoded_bits = len(encoded_bitstream)
    total_bits = num_encoded_bits * REPETITION_FACTOR

    # 4. Siapkan Spread Spectrum (Modulated Bits)
    spatial_key = generate_spatial_key(password, total_bits)
    repeated_bits = np.repeat(encoded_bitstream, REPETITION_FACTOR)
    bipolar_bits = np.where(repeated_bits == 0, -1, 1)
    modulated_bits = bipolar_bits * spatial_key

    # 5. Penyisipan ke sub-band (mendukung mode DUAL)
    if DWT_TARGET_SUBBAND == "DUAL":
        # ======================================================
        # MODE DUAL: Sisipkan ke LL dan HL secara SIMULTAN
        # Keduanya menggunakan modulated_bits yang SAMA PERSIS
        # ======================================================
        LL_embedded = _embed_to_band(LL, modulated_bits, password, alpha)
        HL_embedded = _embed_to_band(HL, modulated_bits, password, alpha)
        Y_reconstructed = pywt.idwt2((LL_embedded, (LH, HL_embedded, HH)), 'haar')
    elif DWT_TARGET_SUBBAND == "HL":
        HL_embedded = _embed_to_band(HL, modulated_bits, password, alpha)
        Y_reconstructed = pywt.idwt2((LL, (LH, HL_embedded, HH)), 'haar')
    elif DWT_TARGET_SUBBAND == "LH":
        LH_embedded = _embed_to_band(LH, modulated_bits, password, alpha)
        Y_reconstructed = pywt.idwt2((LL, (LH_embedded, HL, HH)), 'haar')
    else:
        # Default: "LL"
        LL_embedded = _embed_to_band(LL, modulated_bits, password, alpha)
        Y_reconstructed = pywt.idwt2((LL_embedded, (LH, HL, HH)), 'haar')

    # 6. Rekonstruksi frame
    Y_reconstructed = np.clip(Y_reconstructed, 0, 255).astype(np.uint8)
    ycrcb[:, :, 0] = Y_reconstructed
    stego_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return stego_frame


def extract_bitstream(frame: np.ndarray, password: str, alpha: float = SVD_SCALING_FACTOR, num_bits: int = WATERMARK_BITS) -> np.ndarray:
    """
    Mengekstrak watermark dari frame video menggunakan blind extraction SVD + ECC decode.

    Mode ekstraksi dikontrol oleh DWT_TARGET_SUBBAND di config.py:
    - "LL", "HL", "LH" : Ekstraksi dari satu sub-band
    - "DUAL"           : Fail-Over Mechanism — coba LL dulu, jika ECC gagal pindah ke HL
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # DWT Level 1
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')

    # Hitung jumlah bit encoded (setelah ECC encode)
    if USE_ECC and _RS_AVAILABLE:
        num_encoded_bits = num_bits + (ECC_SYMBOLS * 8)  # 32 + 32 = 64 bit
    else:
        num_encoded_bits = num_bits

    if DWT_TARGET_SUBBAND == "DUAL":
        # ======================================================
        # MODE DUAL: Fail-Over Mechanism
        # Langkah 1: Coba ekstrak dari LL (lebih robust untuk Blur/Resize)
        # Langkah 2: Jika ECC gagal, beralih ke HL (lebih robust untuk Brightness)
        # Langkah 3: Jika keduanya gagal, fallback ke Majority Vote dari LL
        # ======================================================
        encoded_from_ll = _extract_from_band(LL, password, alpha, num_encoded_bits)
        decoded_from_ll = _try_ecc_decode_strict(encoded_from_ll)

        if decoded_from_ll is not None:
            # LL berhasil diselamatkan oleh ECC
            return decoded_from_ll

        # LL gagal, beralih ke HL
        encoded_from_hl = _extract_from_band(HL, password, alpha, num_encoded_bits)
        decoded_from_hl = _try_ecc_decode_strict(encoded_from_hl)

        if decoded_from_hl is not None:
            # HL berhasil diselamatkan oleh ECC
            return decoded_from_hl

        # Keduanya gagal total (serangan sangat ekstrem) -> fallback dengan decode biasa dari LL
        return _ecc_decode(encoded_from_ll)

    elif DWT_TARGET_SUBBAND == "HL":
        encoded_bits = _extract_from_band(HL, password, alpha, num_encoded_bits)
    elif DWT_TARGET_SUBBAND == "LH":
        encoded_bits = _extract_from_band(LH, password, alpha, num_encoded_bits)
    else:
        # Default: "LL"
        encoded_bits = _extract_from_band(LL, password, alpha, num_encoded_bits)

    # ECC Decode untuk mode non-DUAL
    final_bits = _ecc_decode(encoded_bits)
    return final_bits

def calculate_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) antara dua frame.
    Semakin tinggi nilai PSNR, semakin mirip frame stego dengan frame asli.
    Nilai di atas 35 dB dianggap sangat bagus (kasat mata identik).

    Args:
        original: Frame asli.
        stego: Frame stego.

    Returns:
        Nilai PSNR dalam dB.
    """
    orig = original.astype(np.float64)
    steg = stego.astype(np.float64)
    mse = np.mean((orig - steg) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    Menghitung Bit Error Rate (BER) antara bit asli dan bit yang diekstrak.
    BER = 0.0 artinya sempurna, BER = 1.0 artinya semua bit salah.

    Args:
        original_bits: Bit yang disisipkan.
        extracted_bits: Bit yang berhasil diekstrak.

    Returns:
        Nilai BER (0.0 hingga 1.0).
    """
    min_len = min(len(original_bits), len(extracted_bits))
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    return errors / min_len
