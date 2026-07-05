# =============================================================================
# ml_pipeline/1_generate_dataset.py
# Pembuat Dataset Master: Arsitektur Hibrida (4-Bit Serangan + 1 Skalar Logo)
# Format CSV:
#   - Kolom X (Input)  : x0_* (11) + xw_* (11) + xa_* (11) = 33 kolom fitur
#   - Kolom Y (Target) : y_atk_bit1..4 (4 bit) + y_logo_scalar (1 skalar) = 5 kolom
# Total: 38 kolom per baris
# =============================================================================
import os
import sys
import csv
import time
import glob
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from config import (
    ZODIAK_LABELS, ATTACK_LABELS, TRAINING_PASSWORD,
    DATASET_MASTER_PATH, MASTER_FEATURE_COLUMNS, MASTER_TARGET_COLUMNS,
    VIDEO_FEATURE_COLUMNS, WATERMARK_BITS, VIDEOS_DIR
)
from core.security import generate_zodiak_index, encode_hybrid_labels
from core.watermark_hybrid import embed_bitstream
from core.video_utils import read_video_frames, extract_features
from core.attacks import apply_attack


def find_video_files(video_dir: str) -> list:
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)


def simulate_mp4_compression(frame: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Mensimulasikan artefak kompresi lossy MP4/H.264 menggunakan encoding JPEG.

    Mengapa ini penting:
    Saat Embed di dashboard, video stego disimpan ke disk sbg file .mp4 (lossy),
    lalu dibaca kembali. Proses save+load ini mengubah nilai piksel secara halus
    (misal pixel_mean dari 120.53 -> 120.41). Jika dataset training menggunakan
    array Numpy murni tanpa simulasi ini, AI akan menemui distribusi fitur yang
    BERBEDA dari yang ia pelajari -> menyebabkan prediksi ngawur.

    Args:
        frame: Frame input (BGR Numpy array).
        quality: Kualitas JPEG (75 = mendekati kualitas kompresi MP4 rata-rata).

    Returns:
        Frame setelah melalui encode+decode JPEG (ada artefak kompresi ringan).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", frame, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def process_video(video_path: str):
    """
    Fungsi pekerja (worker) per Core CPU.
    Untuk setiap frame x setiap zodiak x setiap serangan:
      1. Hitung fitur X0 dari frame ASLI
      2. Embed watermark -> frame Stego
         -> Simulasi kompresi MP4 pada Stego (agar distribusi fitur sama dgn production)
      3. Apply serangan -> frame Attacked
         -> Simulasi kompresi MP4 pada Attacked
      4. Encode Target Y HIBRIDA = [4 bit serangan | 1 skalar desimal logo]
      5. Simpan baris [X0 | Xw | Xa | Y_attack (4) | Y_logo_scalar (1)] -> 38 kolom
    """
    video_name = os.path.basename(video_path)
    master_rows_chunk = []

    try:
        frames, fps, w, h = read_video_frames(video_path)
    except Exception as e:
        return video_name, [], f"Gagal membaca video: {e}"

    if len(frames) == 0:
        return video_name, [], "Tidak ada frame yang dibaca."

    # ---- Looping Zodiak & Serangan & Frame ----
    for zodiak_name in ZODIAK_LABELS:
        zodiak_bits = generate_zodiak_index(zodiak_name, WATERMARK_BITS)
        for attack_name in ATTACK_LABELS:
            for frame in frames:
                try:
                    # 1. Hitung fitur X0 (Frame Asli, tanpa kompresi)
                    feats_x0 = extract_features(frame)
                    x0_row = [feats_x0[c] for c in VIDEO_FEATURE_COLUMNS]

                    # 2. Embed watermark -> Frame Stego
                    stego_frame_raw = embed_bitstream(frame, zodiak_bits, TRAINING_PASSWORD)

                    # KUNCI: Simulasi kompresi MP4 agar distribusi fitur Xw
                    # di training SAMA dengan yang dihasilkan saat production (dashboard)
                    stego_frame = simulate_mp4_compression(stego_frame_raw)

                    # 3. Hitung fitur Xw (Frame Stego setelah simulasi kompresi)
                    feats_xw = extract_features(stego_frame)
                    xw_row = [feats_xw[c] for c in VIDEO_FEATURE_COLUMNS]

                    # 4. Apply Serangan -> Frame Attacked
                    attacked_frame_raw = apply_attack(stego_frame, attack_name)

                    # KUNCI: Simulasi kompresi MP4 juga pada frame yang sudah diserang
                    # (kecuali JPEG_Compression karena sudah lossy dari serangan itu sendiri)
                    # FIX: Jika serangan 'Clean', JANGAN lakukan double compression agar Xw dan Xa identik 100%.
                    if attack_name == "Clean":
                        attacked_frame = stego_frame
                    elif attack_name != "JPEG_Compression":
                        attacked_frame = simulate_mp4_compression(attacked_frame_raw)
                    else:
                        attacked_frame = attacked_frame_raw

                    # 5. Hitung fitur Xa (Frame setelah Diserang + kompresi)
                    feats_xa = extract_features(attacked_frame)
                    xa_row = [feats_xa[c] for c in VIDEO_FEATURE_COLUMNS]

                    # 6. Encode Target Y = Hibrida: [4 bit serangan | 1 skalar logo]
                    y_vals = encode_hybrid_labels(attack_name, zodiak_name)

                    # 7. Gabungkan jadi 1 baris: [X0 | Xw | Xa | Y_attack | Y_logo_scalar]
                    master_row = x0_row + xw_row + xa_row + y_vals.tolist()
                    master_rows_chunk.append(master_row)

                except Exception:
                    continue  # Skip frame error

    return video_name, master_rows_chunk, "OK"


def generate_datasets():
    start_time = time.time()
    video_files = find_video_files(VIDEOS_DIR)

    if not video_files:
        print(f"[ERROR] Tidak ada video di folder: {VIDEOS_DIR}")
        return

    print("=" * 70)
    print("GENERATE DATASET MASTER - ARSITEKTUR HIBRIDA")
    print("Format: X0 (11) + Xw (11) + Xa (11) + Y_attack (4-bit) + Y_logo_scalar (1) = 38 Kolom")
    print("=" * 70)
    print(f"Video ditemukan  : {len(video_files)} file")
    print(f"Zodiak           : {len(ZODIAK_LABELS)} kelas")
    print(f"Serangan         : {len(ATTACK_LABELS)} kelas")

    # Header 40 kolom
    master_header = MASTER_FEATURE_COLUMNS + MASTER_TARGET_COLUMNS
    all_master_rows = []

    # Multiprocessing
    max_workers = min(os.cpu_count() or 4, len(video_files), 14)
    print(f"Mengaktifkan {max_workers} Core CPU secara serentak...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_video, path): path for path in video_files}

        for future in concurrent.futures.as_completed(future_to_video):
            v_name, rows, status = future.result()
            if status == "OK":
                all_master_rows.extend(rows)
                print(f"[SELESAI] {v_name} -> {len(rows)} baris berhasil diproses.")
            else:
                print(f"[ERROR] {v_name} -> {status}")

    print(f"\n\n{'=' * 70}")
    print(f"PROSES SELESAI dalam {time.time() - start_time:.1f} detik")
    print(f"Total baris dataset master : {len(all_master_rows)}")

    # Simpan ke CSV
    with open(DATASET_MASTER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(master_header)
        writer.writerows(all_master_rows)

    print(f"[OK] Dataset tersimpan di: {DATASET_MASTER_PATH}")
    print(f"\nVerifikasi Kolom (38 total):")
    print(f"  - Kolom X0        : x0_pixel_mean ... x0_glcm_energy  (11 kolom, angka desimal)")
    print(f"  - Kolom Xw        : xw_pixel_mean ... xw_glcm_energy  (11 kolom, angka desimal)")
    print(f"  - Kolom Xa        : xa_pixel_mean ... xa_glcm_energy  (11 kolom, angka desimal)")
    print(f"  - Kolom Y Serangan: y_atk_bit1 ... y_atk_bit4        (4 kolom, hanya 0 atau 1)")
    print(f"  - Kolom Y Logo    : y_logo_scalar                      (1 kolom, angka 0-7)")
    print(f"\nLanjut ke: python ml_pipeline/2_train_master_model.py")
    print("=" * 70)


if __name__ == "__main__":
    generate_datasets()
