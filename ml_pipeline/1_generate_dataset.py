# =============================================================================
# ml_pipeline/1_generate_dataset.py
# Pembuat Dataset: Mega-Looping dengan Multiprocessing (Paralel)
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
    DATASET_VIDEO_PATH, DATASET_LOGO_PATH,
    VIDEO_FEATURE_COLUMNS, LOGO_FEATURE_COLUMNS,
    WATERMARK_BITS, VIDEOS_DIR
)
from core.security import generate_zodiak_index
from core.watermark_hybrid import embed_bitstream, extract_bitstream
from core.video_utils import read_video_frames, extract_features
from core.attacks import apply_attack

def find_video_files(video_dir: str) -> list:
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)

def process_video(video_path: str):
    """Fungsi pekerja (worker) yang dijalankan oleh masing-masing Core CPU."""
    video_name = os.path.basename(video_path)
    video_rows_chunk = []
    logo_rows_chunk = []
    
    try:
        frames, fps, w, h = read_video_frames(video_path)
    except Exception as e:
        return video_name, [], [], f"Gagal membaca video: {e}"

    if len(frames) == 0:
        return video_name, [], [], "Tidak ada frame yang dibaca."

    # ---- Ambil frame HOST asli ----
    for frame in frames[:20]:
        try:
            feats = extract_features(frame)
            row = [feats[c] for c in VIDEO_FEATURE_COLUMNS]
            row += [0, "N/A", "N/A", video_name]  # label_watermark=0
            video_rows_chunk.append(row)
        except Exception:
            continue

    # ---- Looping Zodiak & Serangan ----
    for zodiak_name in ZODIAK_LABELS:
        zodiak_bits = generate_zodiak_index(zodiak_name, WATERMARK_BITS)
        for attack_name in ATTACK_LABELS:
            for frame in frames:
                try:
                    # 1. Embed
                    stego_frame = embed_bitstream(frame, zodiak_bits, TRAINING_PASSWORD)
                    # 2. Attack
                    attacked_frame = apply_attack(stego_frame, attack_name)
                    # 3. Extract Features
                    video_feats = extract_features(attacked_frame)
                    video_row = [video_feats[c] for c in VIDEO_FEATURE_COLUMNS]
                    video_row += [1, attack_name, zodiak_name, video_name]
                    video_rows_chunk.append(video_row)
                    # 4. Extract Bits
                    extracted_bits = extract_bitstream(attacked_frame, TRAINING_PASSWORD, num_bits=WATERMARK_BITS)
                    logo_row = list(extracted_bits.astype(int)) + [zodiak_name, attack_name, video_name]
                    logo_rows_chunk.append(logo_row)
                except Exception as e:
                    continue  # Skip frame error
                    
    return video_name, video_rows_chunk, logo_rows_chunk, "OK"

def generate_datasets():
    start_time = time.time()
    video_files = find_video_files(VIDEOS_DIR)
    
    if not video_files:
        print(f"[ERROR] Tidak ada video di folder: {VIDEOS_DIR}")
        return

    print("=" * 70)
    print("GENERATE DATASET - MODE PARALEL MULTIPROCESSING")
    print("=" * 70)
    print(f"Video ditemukan  : {len(video_files)} file")
    
    video_header = VIDEO_FEATURE_COLUMNS + ["label_watermark", "label_attack", "zodiak", "video_source"]
    logo_header = LOGO_FEATURE_COLUMNS + ["label_zodiak", "attack_applied", "video_source"]

    all_video_rows = []
    all_logo_rows = []

    # Membuka gerbang Multiprocessing (Limit CPU Core)
    max_workers = min(os.cpu_count() or 4, len(video_files), 14)
    print(f"Mengaktifkan {max_workers} Core CPU secara serentak...\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit semua tugas
        future_to_video = {executor.submit(process_video, path): path for path in video_files}
        
        # Ambil hasil yang sudah selesai
        for future in concurrent.futures.as_completed(future_to_video):
            v_name, v_rows, l_rows, status = future.result()
            if status == "OK":
                all_video_rows.extend(v_rows)
                all_logo_rows.extend(l_rows)
                print(f"[SELESAI] {v_name} -> {len(v_rows)} frame berhasil diproses.")
            else:
                print(f"[ERROR] {v_name} -> {status}")

    print(f"\n\n{'=' * 70}")
    print(f"PROSES SELESAI dalam {time.time() - start_time:.1f} detik")
    print(f"Total baris dataset video  : {len(all_video_rows)}")
    print(f"Total baris dataset logo   : {len(all_logo_rows)}")

    # ---- Simpan ke CSV ----
    with open(DATASET_VIDEO_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(video_header)
        writer.writerows(all_video_rows)

    with open(DATASET_LOGO_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(logo_header)
        writer.writerows(all_logo_rows)

    print(f"[OK] Dataset tersimpan di: data/dataset_video.csv & data/dataset_logo.csv")

if __name__ == "__main__":
    generate_datasets()
