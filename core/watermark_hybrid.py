# =============================================================================
# core/watermark_hybrid.py
# Implementasi Hybrid DWT-DCT-SVD dengan Spread Spectrum
# =============================================================================

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from config import DCT_BLOCK_SIZE, SVD_SCALING_FACTOR, REPETITION_FACTOR
from core.security import generate_spatial_key, _password_to_seed

def _get_shuffled_blocks(password: str, blocks_h: int, blocks_w: int, total_bits: int):
    all_blocks = [(i, j) for i in range(blocks_h) for j in range(blocks_w)]
    seed_int = _password_to_seed(password)
    rng = np.random.RandomState(seed_int)
    rng.shuffle(all_blocks)
    return all_blocks[:total_bits]


def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def embed_bitstream(frame: np.ndarray, bitstream: np.ndarray, password: str, alpha: float = SVD_SCALING_FACTOR) -> np.ndarray:
    """
    Menyisipkan 64-bit watermark ke dalam frame video menggunakan DWT-DCT-SVD-SS.
    """
    # 1. Konversi ke YCrCb dan ambil channel Luminance (Y)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # 2. DWT Level 1
    coeffs2 = pywt.dwt2(Y, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # 3. Siapkan Spread Spectrum Bits
    num_bits = len(bitstream)
    total_bits = num_bits * REPETITION_FACTOR
    spatial_key = generate_spatial_key(password, total_bits)  # Bipolar +1/-1

    repeated_bits = np.repeat(bitstream, REPETITION_FACTOR)
    bipolar_bits = np.where(repeated_bits == 0, -1, 1)
    modulated_bits = bipolar_bits * spatial_key

    # 4. Pecah sub-band LL menjadi blok-blok DCT_BLOCK_SIZE
    h, w = LL.shape
    blocks_h = h // DCT_BLOCK_SIZE
    blocks_w = w // DCT_BLOCK_SIZE

    if blocks_h * blocks_w < total_bits:
        raise ValueError(f"Ukuran frame terlalu kecil untuk memuat {total_bits} bit.")

    LL_embedded = LL.copy()
    
    selected_blocks = _get_shuffled_blocks(password, blocks_h, blocks_w, total_bits)

    for bit_idx, (i, j) in enumerate(selected_blocks):
        r_start, r_end = i * DCT_BLOCK_SIZE, (i + 1) * DCT_BLOCK_SIZE
        c_start, c_end = j * DCT_BLOCK_SIZE, (j + 1) * DCT_BLOCK_SIZE
        block = LL_embedded[r_start:r_end, c_start:c_end]

        # DCT
        dct_block = apply_dct(block)

        # SVD
        U, S, V = np.linalg.svd(dct_block, full_matrices=False)

        # Kuantisasi STDM (QIM) pada Singular Value terbesar (S[0])
        m = 0 if modulated_bits[bit_idx] == -1 else 1
        # Rumus kuantisasi blind
        quantized = np.round((S[0] - m * (alpha / 2)) / alpha) * alpha + m * (alpha / 2)
        S[0] = quantized

        # Rekonstruksi SVD -> IDCT
        dct_reconstructed = np.dot(U, np.dot(np.diag(S), V))
        block_reconstructed = apply_idct(dct_reconstructed)
        
        LL_embedded[r_start:r_end, c_start:c_end] = block_reconstructed

    # 5. Inverse DWT
    Y_reconstructed = pywt.idwt2((LL_embedded, (LH, HL, HH)), 'haar')
    Y_reconstructed = np.clip(Y_reconstructed, 0, 255).astype(np.uint8)

    # Gabungkan kembali ke BGR
    ycrcb[:, :, 0] = Y_reconstructed
    stego_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return stego_frame


def extract_bitstream(frame: np.ndarray, password: str, alpha: float = SVD_SCALING_FACTOR, num_bits: int = 64) -> np.ndarray:
    """
    Mengekstrak 64-bit watermark dari frame video menggunakan blind extraction SVD.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # DWT Level 1
    coeffs2 = pywt.dwt2(Y, 'haar')
    LL, _ = coeffs2

    total_bits = num_bits * REPETITION_FACTOR
    spatial_key = generate_spatial_key(password, total_bits)

    h, w = LL.shape
    blocks_h = h // DCT_BLOCK_SIZE
    blocks_w = w // DCT_BLOCK_SIZE

    extracted_modulated = []
    selected_blocks = _get_shuffled_blocks(password, blocks_h, blocks_w, total_bits)

    for bit_idx, (i, j) in enumerate(selected_blocks):
        r_start, r_end = i * DCT_BLOCK_SIZE, (i + 1) * DCT_BLOCK_SIZE
        c_start, c_end = j * DCT_BLOCK_SIZE, (j + 1) * DCT_BLOCK_SIZE
        block = LL[r_start:r_end, c_start:c_end]

        # DCT
        dct_block = apply_dct(block)
        
        # SVD
        U, S, V = np.linalg.svd(dct_block, full_matrices=False)

        val = S[0]
        
        # Jarak Kuantisasi ke titik bit '0' dan bit '1'
        dist_0 = abs(val - (np.round(val / alpha) * alpha))
        dist_1 = abs(val - (np.round((val - alpha / 2) / alpha) * alpha + alpha / 2))

        if dist_1 < dist_0:
            extracted_modulated.append(1)
        else:
            extracted_modulated.append(-1)

    extracted_modulated = np.array(extracted_modulated)
    
    # Demodulasi dengan Spread Spectrum Vector
    demodulated = extracted_modulated * spatial_key

    # Aggregate per repetition (Majority Vote)
    demodulated_reshaped = demodulated.reshape((num_bits, REPETITION_FACTOR))
    sums = np.sum(demodulated_reshaped, axis=1)
    
    final_bits = np.where(sums > 0, 1, 0).astype(np.uint8)
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
