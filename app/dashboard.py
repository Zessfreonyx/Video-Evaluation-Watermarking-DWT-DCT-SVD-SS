# =============================================================================
# app/dashboard.py
# Dashboard Streamlit: Testkit Sistem Watermarking Berbasis Indeks DCT-STDM
# =============================================================================
#
# CARA PAKAI:
#   streamlit run app/dashboard.py
#   (dari direktori root proyek)
#
# =============================================================================

import os
import sys
import json
import pickle
import time
import datetime
import uuid
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
import pandas as pd

from config import (
    ZODIAK_LABELS, ATTACK_LABELS, WATERMARK_BITS,
    SVD_SCALING_FACTOR,
    MODEL_DETECTOR_PATH, MODEL_ATTACK_PATH, MODEL_LOGO_PATH, MODEL_MASTER_PATH,
    SESSIONS_DIR, REPORTS_DIR, OUTPUT_DIR, TRAINING_PASSWORD,
    VIDEO_FEATURE_COLUMNS, LOGO_FEATURE_COLUMNS, MASTER_FEATURE_COLUMNS,
    MASTER_TARGET_COLUMNS, LOGOS_DIR,
)
from core.security import (
    generate_zodiak_index, generate_temporal_key, verify_password_strength,
    decode_hybrid_labels, attack_to_4bit, zodiak_to_3bit
)
from core.watermark_hybrid import embed_bitstream, extract_bitstream, calculate_psnr, calculate_ssim, calculate_ber
from core.video_utils import read_all_frames, save_video, extract_features, get_video_info
from core.attacks import apply_attack


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Watermark Zodiak - Testkit",
    page_icon="♈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #7C3AED; margin-bottom: 0; }
    .sub-header  { font-size: 1.05rem; color: #6B7280; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .metric-val  { font-size: 2.4rem; font-weight: 800; color: #A78BFA; }
    .metric-lbl  { font-size: 0.85rem; color: #C4B5FD; text-transform: uppercase; letter-spacing: 1px; }
    .result-box  {
        background: #F0FDF4; border: 2px solid #22C55E; border-radius: 10px;
        padding: 16px; margin: 8px 0; color: #166534;
    }
    .warning-box {
        background: #FFF7ED; border: 2px solid #F97316; border-radius: 10px;
        padding: 16px; margin: 8px 0; color: #9A3412;
    }
    .error-box {
        background: #FEF2F2; border: 2px solid #EF4444; border-radius: 10px;
        padding: 16px; margin: 8px 0; color: #991B1B;
    }
    .step-badge {
        background: #7C3AED; color: white; border-radius: 20px;
        padding: 4px 14px; font-size: 0.85rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI MUAT MODEL
# ============================================================

@st.cache_resource(show_spinner="Memuat model AI...")
def load_models():
    """Memuat model AI master (skema Pak Gelar). Di-cache agar tidak reload setiap interaksi."""
    models = {}
    if os.path.exists(MODEL_MASTER_PATH):
        with open(MODEL_MASTER_PATH, "rb") as f:
            data = pickle.load(f)
            # Backward compatibility check
            if "models" in data:
                models["master"] = data
            else:
                models["master"] = {
                    "models": {data["best_algorithm"]: data["model"]},
                    "best_algorithm": data["best_algorithm"],
                    "feature_columns": data["feature_columns"],
                    "target_columns": data["target_columns"],
                    "zodiak_labels": data["zodiak_labels"],
                    "attack_labels": data["attack_labels"]
                }
    else:
        models["master"] = None
    # Model lama (backward compatibility, optional)
    for name, path in [
        ("attack", MODEL_ATTACK_PATH),
        ("logo", MODEL_LOGO_PATH),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            models[name] = None
    return models


# ============================================================
# FUNGSI PIPELINE DETEKSI (TEMPORAL AVERAGING)
# ============================================================

def run_master_detection(
    frames_original: list,
    frames_stego: list,
    frames_attacked: list,
    models: dict,
    temporal_key: list,
    num_sample: int = 10,
    selected_algorithm: str = None
):
    """
    Pipeline Deteksi Skema Pak Gelar (Non-Blind, Multi-Output).
    Input: 3 set frame (Original X0, Stego Xw, Attacked Xa)
    Output: dict berisi hasil prediksi 7 bit (Serangan + Zodiak).
    """
    master_bundle = models.get("master")
    if not master_bundle:
        return {"error": "Model Master belum tersedia. Jalankan 2_train_master_model.py terlebih dahulu."}

    # Ambil model berdasarkan pilihan user, atau default ke algoritma terbaik
    if selected_algorithm and "models" in master_bundle and selected_algorithm in master_bundle["models"]:
        ml_model = master_bundle["models"][selected_algorithm]
    else:
        # Fallback
        best_name = master_bundle.get("best_algorithm")
        ml_model = master_bundle["models"].get(best_name) if "models" in master_bundle else master_bundle.get("model")

    total = min(len(frames_original), len(frames_stego), len(frames_attacked))

    # Sample frame HANYA dari yang disisipi watermark (temporal_key)
    if not temporal_key:
        return {"error": "Temporal key kosong. Tidak ada frame bermarka."}
    
    # Ambil sampel acak atau urut dari temporal_key
    if len(temporal_key) <= num_sample:
        indices = temporal_key
    else:
        indices = np.random.choice(temporal_key, num_sample, replace=False).tolist()

    all_y_preds = []  # List of 7-bit arrays

    for idx in indices:
        try:
            feats_x0 = extract_features(frames_original[idx])
            feats_xw = extract_features(frames_stego[idx])
            feats_xa = extract_features(frames_attacked[idx])

            x0_row = [feats_x0[c] for c in VIDEO_FEATURE_COLUMNS]
            xw_row = [feats_xw[c] for c in VIDEO_FEATURE_COLUMNS]
            xa_row = [feats_xa[c] for c in VIDEO_FEATURE_COLUMNS]

            feat_vec = np.array(x0_row + xw_row + xa_row).reshape(1, -1)

            # Prediksi: output shape (1, 7) -> ambil baris pertama
            y_pred_bits = ml_model.predict(feat_vec)[0]  # array 7 bit
            all_y_preds.append(y_pred_bits)

        except Exception:
            continue

    if not all_y_preds:
        return {"error": "Tidak ada frame yang berhasil diproses."}

    # Majority voting:
    # - Untuk 4 bit serangan: majority vote per bit (round of mean)
    # - Untuk 1 skalar logo: mode (nilai yang paling sering muncul)
    y_preds_arr = np.array(all_y_preds)  # shape: (n_frames, 5)
    attack_vote = np.round(np.mean(y_preds_arr[:, :4], axis=0)).astype(int)  # 4 bit
    logo_values, logo_counts = np.unique(y_preds_arr[:, 4].astype(int), return_counts=True)
    logo_vote = int(logo_values[np.argmax(logo_counts)])  # skalar: nilai yang paling sering
    y_final = np.append(attack_vote, logo_vote)

    # Decode 5 nilai ke nama serangan dan zodiak (pakai decoder hibrida)
    attack_result, zodiak_result = decode_hybrid_labels(y_final)

    # Hitung confidence:
    # - Attack Confidence: rata-rata persetujuan per-bit (dijamin >= 50%)
    attack_matches_per_bit = (y_preds_arr[:, :4] == y_final[:4])
    attack_conf = float(np.mean(np.sum(attack_matches_per_bit, axis=0) / len(all_y_preds)))

    # - Logo Confidence: proporsi frame yang memilih logo pemenang
    logo_conf = float(np.sum(y_preds_arr[:, 4].astype(int) == logo_vote) / len(all_y_preds))

    # Hitung Distribusi Voting AI untuk semua Logo
    logo_distribution = {}
    from config import ZODIAK_LABELS
    for val, count in zip(logo_values, logo_counts):
        logo_name = ZODIAK_LABELS[int(val)]
        logo_distribution[logo_name] = float(count / len(all_y_preds))
        
    # Pastikan semua zodiak ada di kamus, isi dengan 0.0 jika tidak ada yang vote
    for name in ZODIAK_LABELS:
        if name not in logo_distribution:
            logo_distribution[name] = 0.0

    return {
        "y_final_bits": y_final.tolist(),
        "attack_result": attack_result,
        "zodiak_result": zodiak_result,
        "attack_confidence": attack_conf,
        "logo_confidence": logo_conf,
        "frames_analyzed": len(all_y_preds),
        "total_frames": total,
        "attack_bits": y_final[:4].tolist(),
        "logo_scalar": logo_vote,
        "logo_distribution": logo_distribution,
    }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.markdown('<p class="main-header">♈ Zodiak<br>Watermark</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Sistem DWT-SVD-SS + AI Master</p>', unsafe_allow_html=True)
    st.sidebar.divider()

    menu = st.sidebar.radio(
        "Navigasi",
        ["🔐 Embed Watermark", "💥 Simulasi Serangan", "🔍 Deteksi & Ekstrak", "📊 Laporan ML", "📂 Folder Output Sesi"],
        label_visibility="collapsed"
    )
    st.sidebar.divider()

    st.sidebar.markdown("**Status Model AI:**")
    master_ready = os.path.exists(MODEL_MASTER_PATH)
    icon_m = "🟢" if master_ready else "🔴"
    st.sidebar.markdown(f"{icon_m} Model Master (Pak Gelar)")
    # Backward compatibility display
    for label, path in [
        ("Spesialis Serangan (Lama)", MODEL_ATTACK_PATH),
        ("Spesialis Logo (Lama)", MODEL_LOGO_PATH),
    ]:
        icon = "🟡" if os.path.exists(path) else "⚫"
        st.sidebar.markdown(f"{icon} {label}")

    return menu




# ============================================================
# HELPER: SCAN PROJECT SESSIONS
# ============================================================

def get_project_sessions():
    """Scan OUTPUT_DIR dan kembalikan semua folder proyek (awalan 'Proj_')."""
    sessions = []
    if os.path.exists(OUTPUT_DIR):
        for d in os.listdir(OUTPUT_DIR):
            proj_path = os.path.join(OUTPUT_DIR, d)
            if d.startswith("Proj_") and os.path.isdir(proj_path):
                sessions.append(d)
    sessions.sort(reverse=True)
    return sessions


def load_project_record(session_name):
    """Baca file record.json dari folder proyek."""
    record_path = os.path.join(OUTPUT_DIR, session_name, "project.json")
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_project_record(session_name, record):
    """Simpan/update file project.json ke folder proyek."""
    record_path = os.path.join(OUTPUT_DIR, session_name, "project.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4, ensure_ascii=False)


# ============================================================
# HALAMAN: EMBED WATERMARK
# ============================================================

def page_embed():
    st.markdown('<h1 class="main-header">🔐 Embed Watermark</h1>', unsafe_allow_html=True)
    st.markdown("Sisipkan indeks logo zodiak ke dalam video host menggunakan Hybrid DWT-DCT-SVD-SS.")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        project_name = st.text_input("📁 Nama Proyek Baru", placeholder="Misal: Video_Porsche_Tes1", help="Semua file sesi ini (Original, Stego, Serangan) akan disimpan di folder ini.")
        uploaded_video = st.file_uploader("Upload Video Host (Asli)", type=["mp4", "avi", "mov"])
        zodiak_choice = st.selectbox("Pilih Logo Zodiak", ZODIAK_LABELS)
        password = st.text_input("Kata Sandi Rahasia", type="password", placeholder="Minimal 6 karakter")
        num_frames_embed = st.selectbox(
            "Jumlah Frame Disisipkan 🛡️",
            options=[50, 100, 150],
            index=0,
            help="Semakin banyak frame yang disisipi, semakin tahan banting terhadap serangan pemotongan video. Saat Ekstraksi, Anda bisa memilih subset frame yang lebih kecil (5/15/30) untuk efisiensi."
        )

    with col2:
        if uploaded_video:
            st.info(f"📹 File: **{uploaded_video.name}** ({uploaded_video.size / 1024:.1f} KB)")
        st.markdown("""
        **Alur Project Workspace:**
        1. ✅ **[HALAMAN INI]** Buat Proyek & Embed watermark
        2. 🔲 Halaman Simulasi Serangan → pilih proyek ini
        3. 🔲 Halaman Deteksi → pilih proyek & video tersangka
        """)

    if st.button("🚀 Mulai Penyisipan Watermark", use_container_width=True, type="primary"):
        if not project_name.strip():
            st.error("❌ Harap isi Nama Proyek terlebih dahulu.")
            return
        if not uploaded_video:
            st.error("❌ Harap upload video.")
            return
        if not verify_password_strength(password):
            st.error("❌ Kata sandi minimal 6 karakter.")
            return

        safe_name = project_name.strip().replace(" ", "_")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"Proj_{safe_name}_{timestamp}"
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        with st.spinner("Memproses video..."):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            try:
                frames, fps, w, h = read_all_frames(tmp_path)
                zodiak_bits = generate_zodiak_index(zodiak_choice, WATERMARK_BITS)
                
                # Gunakan num_frames_embed dari input user
                actual_frames_to_embed = min(num_frames_embed, len(frames))
                temporal_key = generate_temporal_key(password, len(frames), actual_frames_to_embed)

                stego_frames = frames.copy()
                for idx in temporal_key:
                    stego_frames[idx] = embed_bitstream(frames[idx], zodiak_bits, password, alpha=SVD_SCALING_FACTOR)

                sample_idx = temporal_key[0] if temporal_key else 0
                psnr = calculate_psnr(frames[sample_idx], stego_frames[sample_idx])
                ssim_score = calculate_ssim(frames[sample_idx], stego_frames[sample_idx])

                # Simpan Video Stego
                stego_name = "stego.mp4"
                stego_path = os.path.join(session_dir, stego_name)
                save_video(stego_frames, stego_path, fps, w, h)

                # Simpan Video Asli (wajib untuk referensi AI Non-Blind nanti)
                original_name = "original.mp4"
                original_path = os.path.join(session_dir, original_name)
                shutil.copy2(tmp_path, original_path)

                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Terjadi error: {e}")
                return

        st.success(f"✅ Watermark berhasil disisipkan! Proyek tersimpan di: `{session_dir}`")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{psnr:.1f}</div><div class="metric-lbl">PSNR (dB)</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{ssim_score:.4f}</div><div class="metric-lbl">SSIM</div></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(temporal_key)}</div><div class="metric-lbl">Frame Disisipkan</div></div>', unsafe_allow_html=True)
        with col_d:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{WATERMARK_BITS}</div><div class="metric-lbl">Bit Indeks</div></div>', unsafe_allow_html=True)

        # Simpan project.json
        record = {
            "session_id": session_id,
            "project_name": safe_name,
            "created_at": datetime.datetime.now().isoformat(),
            "zodiak": zodiak_choice,
            "psnr": round(psnr, 2),
            "ssim": float(ssim_score),
            "frames_embedded": len(temporal_key),
            "video_file_original": uploaded_video.name,
            "original_path": original_path,
            "stego_path": stego_path,
            "attacked_videos": {},
            "detections": [],
        }
        save_project_record(session_id, record)

        st.info(f"📁 Proyek **{session_id}** siap. Lanjutkan ke **Simulasi Serangan** atau langsung ke **Deteksi & Ekstrak** untuk skenario Clean.")
        with open(stego_path, "rb") as f:
            st.download_button("⬇️ Download Video Stego (Preview)", f, file_name=stego_name, mime="video/mp4")



# ============================================================
# HALAMAN: SIMULASI SERANGAN (PROJECT WORKSPACE)
# ============================================================

def page_attack_simulation():
    st.markdown('<h1 class="main-header">💥 Simulasi Serangan</h1>', unsafe_allow_html=True)
    st.markdown("Pilih proyek yang sudah di-embed, lalu terapkan serangan. Hasil otomatis tersimpan ke folder proyek yang sama.")
    st.divider()

    sessions = get_project_sessions()
    if not sessions:
        st.warning("⚠️ Belum ada proyek tersedia. Buat proyek baru di halaman **Embed Watermark** terlebih dahulu.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_session = st.selectbox("📁 Pilih Proyek", sessions)
        attack_options = [atk for atk in ATTACK_LABELS if atk != "Clean"]
        attack_type = st.selectbox("Pilih Jenis Serangan", attack_options)

    record = load_project_record(selected_session)
    with col2:
        if record:
            st.markdown("**Info Proyek Terpilih:**")
            st.markdown(f"- 🎯 **Logo:** `{record.get('zodiak', '-')}`")
            st.markdown(f"- 📅 **Dibuat:** `{record.get('created_at', '-')[:19]}`")
            existing_attacks = list(record.get("attacked_videos", {}).keys())
            if existing_attacks:
                st.markdown(f"- 💥 **Serangan sudah ada:** {', '.join(existing_attacks)}")
            st.warning(f"Serangan yang akan diterapkan: **{attack_type}**")

    if st.button("💣 Terapkan Serangan ke Proyek!", use_container_width=True, type="primary"):
        if not record:
            st.error("❌ File proyek.json tidak ditemukan. Folder proyek mungkin rusak.")
            return

        stego_path = record.get("stego_path")
        if not stego_path or not os.path.exists(stego_path):
            st.error("❌ File stego.mp4 tidak ditemukan di folder proyek.")
            return

        session_dir = os.path.join(OUTPUT_DIR, selected_session)

        with st.spinner(f"Menerapkan serangan {attack_type} ke video stego..."):
            try:
                frames, fps, w, h = read_all_frames(stego_path)
                attacked_frames = []
                progress_bar = st.progress(0)
                for i, frame in enumerate(frames):
                    attacked_frames.append(apply_attack(frame, attack_type))
                    progress_bar.progress((i + 1) / len(frames))

                output_name = f"attacked_{attack_type}.mp4"
                output_path = os.path.join(session_dir, output_name)
                save_video(attacked_frames, output_path, fps, w, h)

            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Update project.json dengan info serangan baru
        if "attacked_videos" not in record:
            record["attacked_videos"] = {}
        record["attacked_videos"][attack_type] = output_path
        save_project_record(selected_session, record)

        st.success(f"✅ Serangan **{attack_type}** berhasil! Tersimpan di: `{output_path}`")
        st.info("📌 Sekarang buka halaman **Deteksi & Ekstrak** → pilih proyek ini → pilih video tersangka yang baru saja dibuat.")
        with open(output_path, "rb") as f:
            st.download_button(f"⬇️ Download Video Serangan ({attack_type})", f, file_name=output_name, mime="video/mp4")


# ============================================================
# HALAMAN: DETEKSI & EKSTRAK (PROJECT WORKSPACE)
# ============================================================

def page_detect(models: dict):
    st.markdown('<h1 class="main-header">🔍 Deteksi & Ekstrak</h1>', unsafe_allow_html=True)
    st.markdown("Pilih proyek, masukkan sandi, lalu pilih video tersangka. Sistem akan menghitung BER dan memprediksi Serangan & Logo via AI Master.")
    st.divider()

    sessions = get_project_sessions()
    if not sessions:
        st.warning("⚠️ Belum ada proyek tersedia. Buat proyek baru di halaman **Embed Watermark** terlebih dahulu.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**1. Pilih Proyek & Konfigurasi**")
        selected_session = st.selectbox("📁 Pilih Proyek", sessions)
        password = st.text_input("🔑 Kata Sandi Rahasia", type="password", placeholder="Sandi yang dipakai saat Embed")
        num_frames = st.selectbox(
            "Jumlah Frame Dianalisis ⚡",
            options=[5, 15, 30],
            index=2,
            help="Anda hanya perlu menganalisis SUBSET frame dari keseluruhan frame yang disisipkan. Sistem akan otomatis sinkronisasi dengan kunci embed. 5 = Super Cepat, 30 = Lebih Teliti."
        )
        
        master_data = models.get("master")
        available_models = list(master_data["models"].keys()) if master_data and "models" in master_data else []
        selected_algorithm = None
        if available_models:
            default_idx = available_models.index(master_data["best_algorithm"]) if master_data["best_algorithm"] in available_models else 0
            selected_algorithm = st.selectbox("🤖 Pilih Algoritma AI (Multi-Model):", available_models, index=default_idx)

    record = load_project_record(selected_session)

    with col2:
        st.markdown("**2. Pilih Video Tersangka (Xa)**")
        if record:
            # Bangun daftar pilihan video di dalam proyek
            video_choices = {}
            stego_path = record.get("stego_path", "")
            if stego_path and os.path.exists(stego_path):
                video_choices["stego.mp4 (Belum diserang / Skenario Clean)"] = stego_path
            for atk_name, atk_path in record.get("attacked_videos", {}).items():
                if os.path.exists(atk_path):
                    video_choices[f"attacked_{atk_name}.mp4 (Serangan: {atk_name})"] = atk_path

            if video_choices:
                selected_video_label = st.selectbox("Pilih Video Tersangka", list(video_choices.keys()))
                selected_video_path = video_choices[selected_video_label]
                st.info(f"**Proyek:** `{record.get('project_name', '-')}`\n\n**Logo Asli:** `{record.get('zodiak', '-')}`\n\n**Video dipilih:** `{os.path.basename(selected_video_path)}`")
            else:
                st.error("Tidak ada video tersedia di proyek ini. Cek apakah file stego atau attacked masih ada.")
                return
        else:
            st.warning("Proyek tidak valid.")
            return

    master_ready = models.get("master") is not None
    if not master_ready:
        st.warning("⚠️ Model Master belum tersedia. Jalankan: `python ml_pipeline/2_train_master_model.py`")

    if st.button("🔬 Mulai Ekstraksi BER & Analisis AI Master", use_container_width=True, type="primary"):
        if not password:
            st.error("❌ Harap masukkan Kata Sandi.")
            return

        orig_path = record.get("original_path")
        stego_path = record.get("stego_path")
        zodiak_true = record.get("zodiak")

        if not orig_path or not os.path.exists(orig_path):
            st.error("❌ File original.mp4 tidak ditemukan di proyek ini.")
            return
        if not stego_path or not os.path.exists(stego_path):
            st.error("❌ File stego.mp4 tidak ditemukan di proyek ini.")
            return

        with st.spinner("Membaca frame dari 3 sumber video & menjalankan AI Master..."):
            try:
                frames_x0, _, _, _ = read_all_frames(orig_path)
                frames_xw, _, _, _ = read_all_frames(stego_path)
                frames_xa, _, _, _ = read_all_frames(selected_video_path)

                # =============================================
                # TAHAP 1: VERIFIKASI KRIPTOGRAFI (BER)
                # =============================================
                st.markdown("---")
                st.markdown("### 🧮 Tahap 1: Verifikasi Kriptografi (BER)")

                true_bits = generate_zodiak_index(zodiak_true, WATERMARK_BITS)
                # Bangkitkan ulang SELURUH kunci embed asli menggunakan frames_embedded dari project.json
                # Ini memastikan frame yang diekstrak adalah SUBSET VALID dari frame yang disisipkan
                frames_embedded_original = record.get("frames_embedded", num_frames)
                full_temporal_key = generate_temporal_key(password, len(frames_xa), frames_embedded_original)
                # Ambil hanya num_frames pertama sebagai target ekstraksi (efisiensi)
                temporal_key = full_temporal_key[:num_frames]

                best_math_zodiak = "Unknown"
                best_math_ber = 1.0

                if not temporal_key:
                    ber = 1.0
                else:
                    # --- MAJORITY VOTING BER ---
                    # Ambil sejumlah frame sesuai pilihan num_frames
                    ber_indices = temporal_key[:num_frames]
                    all_extracted_bits = []

                    for idx in ber_indices:
                        bits = extract_bitstream(frames_xa[idx], password=password, alpha=SVD_SCALING_FACTOR, num_bits=WATERMARK_BITS)
                        all_extracted_bits.append(bits)

                    # Majority Vote: jika rata-rata >= 0.5, bit dinyatakan '1', sebaliknya '0'
                    all_extracted_arr = np.array(all_extracted_bits)
                    majority_bits = np.where(np.mean(all_extracted_arr, axis=0) >= 0.5, 1, 0).astype(np.uint8)

                    # Hitung BER terhadap Ground Truth
                    ber = calculate_ber(true_bits, majority_bits)
                    
                    # Hitung BER terhadap SEMUA kemungkinan zodiak untuk mencari "Tebakan Murni Matematika"
                    for z_label in ZODIAK_LABELS:
                        z_bits = generate_zodiak_index(z_label, WATERMARK_BITS)
                        z_ber = calculate_ber(z_bits, majority_bits)
                        if z_ber < best_math_ber:
                            best_math_ber = z_ber
                            best_math_zodiak = z_label

                ber_col1, ber_col2 = st.columns([1, 2])
                with ber_col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{ber:.4f}</div><div class="metric-lbl">Bit Error Rate (BER)</div></div>', unsafe_allow_html=True)
                with ber_col2:
                    if ber < 0.15:
                        st.success(f"✅ **Sandi Valid!** Watermark terdeteksi secara matematis. Cocok dengan profil logo '{zodiak_true}'.")
                    elif ber < 0.35:
                        st.warning("⚠️ **BER Sedang.** Video mungkin mengalami serangan berat. Sandi kemungkinan benar.")
                    else:
                        st.error("❌ **BER Tinggi!** Kemungkinan sandi salah atau watermark hancur.")
                        
                st.info(f"🤖 **Tebakan Murni Matematika:** {best_math_zodiak} (BER terendah: {best_math_ber:.4f})")

                # =============================================
                # TAHAP 2: PREDIKSI AI MASTER
                # =============================================
                result = run_master_detection(frames_x0, frames_xw, frames_xa, models, temporal_key, num_frames, selected_algorithm)

            except Exception as e:
                st.error(f"Error: {e}")
                return

        if "error" in result:
            st.error(result["error"])
            return

        # =============================================
        # ASYMMETRIC TRUST MODEL (Koreksi Cerdas)
        # Kriptografi adalah ilmu pasti, AI adalah probabilitas.
        # Jika Kriptografi (Tahap 1) berhasil memvalidasi watermark (BER < 0.15),
        # maka kita TIMPA tebakan AI yang mungkin meleset dengan fakta matematis.
        # =============================================
        # TRUE HYBRID: RELIABILITY-WEIGHTED FUSION
        # Kriptografi 55%, AI 45% (Dikalibrasi berdasarkan Rapor Akurasi)
        # =============================================
        W_MATH = 0.55
        W_AI_BASE = 0.45
        
        # 1. Hitung Keyakinan Matematika (Drop jika BER naik, hancur di BER >= 0.5)
        # KITA MENGGUNAKAN TEBAKAN MURNI MATEMATIKA, BUKAN GROUND TRUTH
        c_math = max(0.0, 1.0 - (best_math_ber * 2.0))
        math_score = W_MATH * c_math
        
        # 2. Kalibrasi Hak Suara AI berdasarkan "Rapor" Historisnya
        ai_historical_acc = 1.0
        if "results" in master_data and selected_algorithm in master_data["results"]:
            # Ambil akurasi logo spesifik milik algoritma ini
            ai_historical_acc = master_data["results"][selected_algorithm].get("logo_acc", 1.0)
            
        # Hak suara AI yang sesungguhnya (setelah dipotong penalti)
        W_AI_ACTUAL = W_AI_BASE * ai_historical_acc
        
        # 3. Penggabungan Suara (FULL DISTRIBUTION FUSION)
        # Inisialisasi poin 0 untuk semua zodiak
        scores = {z_name: 0.0 for z_name in ZODIAK_LABELS}
            
        # Tambahkan poin dari Kriptografi Matematika ke Pemenang Matematika
        if best_math_zodiak in scores:
            scores[best_math_zodiak] += math_score
        
        # Tambahkan poin dari SELURUH distribusi AI (Bukan cuma pemenang pertamanya saja)
        if "logo_distribution" in result:
            for z_name, z_pct in result["logo_distribution"].items():
                if z_name in scores:
                    scores[z_name] += W_AI_ACTUAL * z_pct
                    
        # 4. Cari Pemenang (Logo dengan Poin Tertinggi)
        winner_logo = max(scores, key=scores.get)
        final_confidence = scores[winner_logo]
        
        # Timpa hasil prediksi AI murni dengan hasil Hibrida ini
        result["zodiak_result"] = winner_logo
        result["logo_confidence"] = final_confidence
        
        # Sesuaikan metadata prediksi agar cocok dengan label kelas
        if winner_logo in ZODIAK_LABELS:
            result["logo_scalar"] = ZODIAK_LABELS.index(winner_logo)
            result["y_final_bits"][4] = result["logo_scalar"]

        # ===== TAMPILKAN HASIL AI =====
        st.markdown("---")
        st.markdown("### 🤖 Tahap 2: Hasil Investigasi AI Master")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{result["attack_result"]}</h2>
                <p>JENIS SERANGAN TERDETEKSI</p>
                <p style="font-size:12px; color:#a0a0a0;">Keyakinan Serangan: {result["attack_confidence"]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{result["zodiak_result"]}</h2>
                <p>LOGO ZODIAK TERDETEKSI</p>
                <p style="font-size:12px; color:#a0a0a0;">Keyakinan Logo: {result["logo_confidence"]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            overall_avg = (result["attack_confidence"] + result["logo_confidence"]) / 2
            st.markdown(f"""
            <div class="metric-box">
                <h2>{overall_avg*100:.1f}%</h2>
                <p>RATA-RATA VOTING KESELURUHAN</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        atk = result["attack_result"]
        st.markdown('<span class="step-badge">HASIL SERANGAN</span>', unsafe_allow_html=True)
        if atk == "Clean":
            st.markdown(f'<div class="result-box">🛡️ <b>Video Bersih</b> — Tidak ada serangan yang terdeteksi.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ <b>Serangan Terdeteksi: {atk}</b></div>', unsafe_allow_html=True)

        logo = result["zodiak_result"]
        st.markdown('<span class="step-badge">HASIL LOGO ZODIAK</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">🔮 <b>Logo Zodiak: {logo}</b> (Keyakinan: {result["logo_confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)
        
        # --- Tampilkan Distribusi Voting AI ---
        if "logo_distribution" in result:
            with st.expander("📊 Lihat Detail Kebingungan AI (Distribusi Suara)"):
                st.markdown('<p style="font-size:14px; color:#a0a0a0;">Berikut adalah rincian tebakan murni AI sebelum dikoreksi oleh Matematika. Semakin merata suaranya, semakin buta AI tersebut akibat kompresi H.264.</p>', unsafe_allow_html=True)
                dist_data = result["logo_distribution"]
                sorted_dist = dict(sorted(dist_data.items(), key=lambda item: item[1], reverse=True))
                
                for z_name, z_pct in sorted_dist.items():
                    if z_pct > 0:
                        st.caption(f"{z_name} ({z_pct*100:.1f}%)")
                        st.progress(float(z_pct))
        # --------------------------------------

        # Visualisasi 5 Nilai Prediksi (Hibrida)
        st.markdown("---")
        st.markdown("**📊 Representasi Hibrida Prediksi AI (Format Target Y):**")
        def format_bit(b):
            color = "#22C55E" if b == 1 else "#EF4444"
            return f'<span style="font-size:1.5rem;font-weight:bold;color:{color}">{b}</span>'

        bit_col1, bit_col2 = st.columns(2)
        with bit_col1:
            atk_bits = result["attack_bits"]
            st.markdown("**Serangan (4 bit):**")
            st.markdown(" | ".join([format_bit(b) for b in atk_bits]), unsafe_allow_html=True)
            st.caption(f"y_atk_bit1={atk_bits[0]}, y_atk_bit2={atk_bits[1]}, y_atk_bit3={atk_bits[2]}, y_atk_bit4={atk_bits[3]}")
        with bit_col2:
            logo_scalar = result["logo_scalar"]
            st.markdown("**Logo Zodiak (Skalar):**")
            st.markdown(f'<span style="font-size:1.5rem;font-weight:bold;color:#3B82F6">{logo_scalar}</span>', unsafe_allow_html=True)
            st.caption(f"y_logo_scalar = {logo_scalar} (Indeks untuk {logo})")

        img_col, info_col = st.columns([1, 2])
        with img_col:
            logo_path_img = os.path.join(LOGOS_DIR, f"{logo}.png")
            if os.path.exists(logo_path_img):
                st.image(logo_path_img, caption=f"Identitas Terverifikasi: {logo}", use_container_width=True)
        with info_col:
            st.markdown(f"**Frame Dianalisis:** {result['frames_analyzed']} dari {result['total_frames']} frame")
            st.markdown(f"**5 Nilai Prediksi (Hibrida):** `{' | '.join(map(str, result['y_final_bits']))}` = `[{' '.join(map(str, result['attack_bits']))}]` (Serangan) + `[{result['logo_scalar']}]` (Skalar Logo)")
            st.markdown(f"**BER:** `{ber:.4f}` | **Video:** `{os.path.basename(selected_video_path)}`")

        # Update project.json dengan hasil deteksi
        detection_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "video_tested": os.path.basename(selected_video_path),
            "ber": round(ber, 4),
            "attack_result": atk,
            "zodiak_result": logo,
            "y_final_bits": result["y_final_bits"],
            "overall_confidence": round(result["logo_confidence"] * 100, 1),
        }
        if "detections" not in record:
            record["detections"] = []
        record["detections"].append(detection_record)
        save_project_record(selected_session, record)
        st.success(f"📂 Hasil deteksi otomatis tersimpan ke proyek: `{selected_session}`")


# ============================================================
# HALAMAN: LAPORAN ML
# ============================================================

def page_reports():
    st.markdown('<h1 class="main-header">📊 Laporan Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown("Hasil evaluasi dari Model AI Master (Skema Pak Gelar: 4 Algoritma, Target Y Desimal).")
    st.divider()

    report_files = {
        "Model Master (Pak Gelar)": "report_master.txt",
        "Model Serangan (Lama)": "report_attack.txt",
        "Model Logo (Lama)": "report_logo.txt",
    }
    cm_files = {
        "Master - Komparasi": "comparison_master.png",
        "Master - Zodiak per Serangan": "master_zodiak_per_attack.png",
        "Serangan (Lama)": "confusion_matrix_attack.png",
        "Logo (Lama)": "confusion_matrix_logo.png",
    }

    tabs = st.tabs(list(report_files.keys()))
    for tab, (title, rfile) in zip(tabs, report_files.items()):
        with tab:
            rpath = os.path.join(REPORTS_DIR, rfile)
            if os.path.exists(rpath):
                with open(rpath, "r", encoding="utf-8") as f:
                    content = f.read()
                st.code(content, language="text")
                with open(rpath, "rb") as f:
                    st.download_button(f"⬇️ Download {rfile}", f, file_name=rfile, mime="text/plain")
            else:
                st.warning(f"Laporan belum tersedia. Jalankan training terlebih dahulu.")

    st.divider()
    st.markdown("### Grafik Komparasi & Confusion Matrix")
    cm_cols = st.columns(2)
    cm_items = list(cm_files.items())
    for i, col in enumerate(cm_cols):
        for j in range(2):
            idx = i * 2 + j
            if idx < len(cm_items):
                name, cfile = cm_items[idx]
                cpath = os.path.join(REPORTS_DIR, cfile)
                with col:
                    if os.path.exists(cpath):
                        st.image(cpath, caption=name, use_container_width=True)
                    else:
                        st.info(f"{name}: Belum ada grafik.")


# ============================================================
# HALAMAN: FOLDER OUTPUT SESI
# ============================================================

def page_history():
    st.markdown('<h1 class="main-header">📂 Folder Output Sesi</h1>', unsafe_allow_html=True)
    st.markdown("Semua riwayat Embed, Serangan, dan Deteksi tersimpan rapi dalam satu folder proyek.")
    st.divider()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    session_folders = [f for f in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
    session_folders.sort(reverse=True)

    if not session_folders:
        st.info("Belum ada sesi tersimpan.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_session = st.selectbox("📁 Pilih Folder Sesi:", session_folders)
        session_path = os.path.join(OUTPUT_DIR, selected_session)
        st.markdown("### Isi Folder:")
        files_in_folder = os.listdir(session_path)
        for f in sorted(files_in_folder):
            icon = "🎬" if f.endswith(('.mp4', '.avi')) else "📄"
            st.markdown(f"- {icon} `{f}`")

    with col2:
        st.markdown(f"### 📋 Rincian: `{selected_session}`")

        # Coba baca project.json (format baru) atau report.json (format lama)
        proj_file = os.path.join(session_path, "project.json")
        report_file = os.path.join(session_path, "report.json")

        record = None
        if os.path.exists(proj_file):
            with open(proj_file, "r", encoding="utf-8") as f:
                record = json.load(f)
        elif os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                record = json.load(f)

        if record:
            # Tampilkan info utama proyek
            if "project_name" in record:
                st.markdown(f"**🏷️ Nama Proyek:** `{record.get('project_name', '-')}`")
                st.markdown(f"**🎯 Logo Zodiak:** `{record.get('zodiak', '-')}`")
                st.markdown(f"**📅 Dibuat:** `{record.get('created_at', '-')[:19]}`")
                st.markdown(f"**📐 PSNR:** `{record.get('psnr', '-')} dB`")
                if "ssim" in record:
                    st.markdown(f"**🔬 SSIM:** `{record.get('ssim', '-')}`")

                # Tampilkan serangan yang pernah dilakukan
                attacked = record.get("attacked_videos", {})
                if attacked:
                    st.markdown(f"**💥 Serangan dalam proyek:** {', '.join(attacked.keys())}")

                # Tampilkan riwayat deteksi
                detections = record.get("detections", [])
                if detections:
                    st.markdown("**🔍 Riwayat Deteksi:**")
                    for i, det in enumerate(detections, 1):
                        with st.expander(f"Deteksi #{i} — {det.get('timestamp', '')[:19]}"):
                            st.markdown(f"- **Video:** `{det.get('video_tested', '-')}`")
                            st.markdown(f"- **BER:** `{det.get('ber', '-')}`")
                            st.markdown(f"- **Serangan:** `{det.get('attack_result', '-')}`")
                            st.markdown(f"- **Logo:** `{det.get('zodiak_result', '-')}`")
                            st.markdown(f"- **Keyakinan:** `{det.get('overall_confidence', '-')}%`")
                            bits = det.get("y_final_bits", [])
                            if bits:
                                st.markdown(f"- **7 Bit:** `{''.join(map(str, bits))}`")
            else:
                st.json(record)

            # Tombol download video yang ada
            video_files = [f for f in files_in_folder if f.endswith(('.avi', '.mp4'))]
            if video_files:
                st.markdown("---")
                st.markdown("**⬇️ Download Video:**")
                for vf in video_files:
                    vpath = os.path.join(session_path, vf)
                    with open(vpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {vf}",
                            data=f,
                            file_name=vf,
                            mime="video/mp4"
                        )
        else:
            st.warning("File project.json / report.json tidak ditemukan di dalam folder ini.")


# ============================================================
# MAIN
# ============================================================

def main():
    models = load_models()
    menu = render_sidebar()

    if menu == "🔐 Embed Watermark":
        page_embed()
    elif menu == "💥 Simulasi Serangan":
        page_attack_simulation()
    elif menu == "🔍 Deteksi & Ekstrak":
        page_detect(models)
    elif menu == "📊 Laporan ML":
        page_reports()
    elif menu == "📂 Folder Output Sesi":
        page_history()


if __name__ == "__main__":
    main()
