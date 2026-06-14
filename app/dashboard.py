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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
import pandas as pd

from config import (
    ZODIAK_LABELS, ATTACK_LABELS, WATERMARK_BITS,
    SVD_SCALING_FACTOR,
    MODEL_DETECTOR_PATH, MODEL_ATTACK_PATH, MODEL_LOGO_PATH,
    SESSIONS_DIR, REPORTS_DIR, OUTPUT_DIR, TRAINING_PASSWORD,
    VIDEO_FEATURE_COLUMNS, LOGO_FEATURE_COLUMNS, LOGOS_DIR,
)
from core.security import generate_zodiak_index, generate_temporal_key, verify_password_strength
from core.watermark_hybrid import embed_bitstream, extract_bitstream, calculate_psnr, calculate_ber
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
    """Memuat ketiga model AI. Di-cache agar tidak reload setiap interaksi."""
    models = {}
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

def run_detection_pipeline(frames: list, password: str, models: dict, num_sample: int = 10, alpha: float = SVD_SCALING_FACTOR):
    """
    Menjalankan pipeline deteksi Dual-AI dengan BER Thresholding.
    Returns: dict berisi hasil akhir deteksi.
    """
    total = len(frames)
    indices = np.linspace(0, total - 1, min(num_sample, total), dtype=int).tolist()
    sampled_frames = [frames[i] for i in indices]

    proba_attack   = []
    proba_logo     = []
    extracted_bits_list = []

    for frame in sampled_frames:
        try:
            feats = extract_features(frame)
            feat_vec = np.array([feats[c] for c in VIDEO_FEATURE_COLUMNS]).reshape(1, -1)

            if models.get("attack"):
                p = models["attack"].predict_proba(feat_vec)[0]
                proba_attack.append(p)

            if models.get("logo"):
                bits = extract_bitstream(frame, password, alpha=alpha, num_bits=WATERMARK_BITS)
                bit_vec = bits.reshape(1, -1).astype(np.float64)
                p = models["logo"].predict_proba(bit_vec)[0]
                proba_logo.append(p)
                extracted_bits_list.append(bits)
        except Exception:
            continue

    if not proba_logo:
        return {"error": "Tidak ada frame yang berhasil diproses."}

    # ---- 1. Prediksi Logo Terlebih Dahulu ----
    logo_result = "Tidak Dikenali"
    logo_conf = 0.0
    all_logo_conf = {}
    
    avg_logo = np.mean(proba_logo, axis=0)
    best_idx = int(np.argmax(avg_logo))
    logo_classes = models["logo"].classes_
    logo_result = logo_classes[best_idx]
    logo_conf = float(avg_logo[best_idx])
    all_logo_conf = {str(logo_classes[i]): float(avg_logo[i]) for i in range(len(logo_classes))}

    # ---- 2. BER Thresholding (Pengganti Model 1) ----
    # Hitung rata-rata bits yang terekstrak
    avg_bits = np.round(np.mean(extracted_bits_list, axis=0)).astype(int)
    
    # Ambil kunci bit asli dari logo yang ditebak
    original_bits = generate_zodiak_index(logo_result, WATERMARK_BITS)
    
    # Hitung Bit Error Rate
    ber_value = calculate_ber(avg_bits, original_bits)
    
    # THRESHOLD = 0.30 (30% toleransi error bit)
    has_watermark = bool(ber_value <= 0.30)

    # ---- 3. Prediksi Serangan ----
    attack_result = "N/A"
    attack_conf = 0.0
    if proba_attack and models.get("attack"):
        avg_attack = np.mean(proba_attack, axis=0)
        best_idx_atk = int(np.argmax(avg_attack))
        attack_classes = models["attack"].classes_
        attack_result = attack_classes[best_idx_atk]
        attack_conf = float(avg_attack[best_idx_atk])

    return {
        "has_watermark": has_watermark,
        "ber_value": float(ber_value),
        "attack_result": attack_result,
        "attack_confidence": attack_conf,
        "logo_result": logo_result,
        "logo_confidence": logo_conf,
        "all_logo_confidence": all_logo_conf,
        "frames_analyzed": len(sampled_frames),
        "total_frames": total,
    }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.markdown('<p class="main-header">♈ Zodiak<br>Watermark</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Sistem DWT-SVD-SS + Triple-AI</p>', unsafe_allow_html=True)
    st.sidebar.divider()

    menu = st.sidebar.radio(
        "Navigasi",
        ["🔐 Embed Watermark", "💥 Simulasi Serangan", "🔍 Deteksi & Ekstrak", "📊 Laporan ML", "📂 Folder Output Sesi"],
        label_visibility="collapsed"
    )
    st.sidebar.divider()

    st.sidebar.markdown("**Status Model AI:**")
    for label, path in [
        ("Spesialis Serangan", MODEL_ATTACK_PATH),
        ("Spesialis Logo", MODEL_LOGO_PATH),
    ]:
        icon = "🟢" if os.path.exists(path) else "🔴"
        st.sidebar.markdown(f"{icon} {label}")

    return menu


# ============================================================
# HALAMAN: EMBED WATERMARK
# ============================================================

def page_embed():
    st.markdown('<h1 class="main-header">🔐 Embed Watermark</h1>', unsafe_allow_html=True)
    st.markdown("Sisipkan indeks logo zodiak ke dalam video host menggunakan Hybrid DWT-DCT-SVD-SS (tahan kompresi MP4).")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_video = st.file_uploader("Upload Video Host", type=["mp4", "avi", "mov"])
        zodiak_choice = st.selectbox("Pilih Logo Zodiak", ZODIAK_LABELS)
        password = st.text_input("Kata Sandi Rahasia", type="password", placeholder="Minimal 6 karakter")

    with col2:
        if uploaded_video:
            st.info(f"📹 File: **{uploaded_video.name}** ({uploaded_video.size / 1024:.1f} KB)")

    if st.button("🚀 Mulai Penyisipan Watermark", use_container_width=True, type="primary"):
        if not uploaded_video:
            st.error("Silakan upload video terlebih dahulu.")
            return
        if not verify_password_strength(password):
            st.error("Kata sandi minimal 6 karakter.")
            return

        # Buat Folder Sesi Unik
        session_id = f"Embed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{zodiak_choice}"
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        with st.spinner("Memproses video..."):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            try:
                frames, fps, w, h = read_all_frames(tmp_path)
                zodiak_bits = generate_zodiak_index(zodiak_choice, WATERMARK_BITS)
                temporal_key = generate_temporal_key(password, len(frames), max(1, len(frames) // 3))

                stego_frames = frames.copy()
                for idx in temporal_key:
                    stego_frames[idx] = embed_bitstream(frames[idx], zodiak_bits, password, alpha=SVD_SCALING_FACTOR)

                sample_idx = temporal_key[0] if temporal_key else 0
                psnr = calculate_psnr(frames[sample_idx], stego_frames[sample_idx])

                # Simpan video stego (Lossy MP4) ke dalam folder sesi
                output_name = f"stego_{zodiak_choice}.mp4"
                output_path = os.path.join(session_dir, output_name)
                save_video(stego_frames, output_path, fps, w, h)

                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Terjadi error: {e}")
                return

        # Tampilkan hasil
        st.success(f"✅ Watermark berhasil disisipkan! Tersimpan di: {session_dir}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{psnr:.1f}</div><div class="metric-lbl">PSNR (dB)</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(temporal_key)}</div><div class="metric-lbl">Frame Disisipkan</div></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{WATERMARK_BITS}</div><div class="metric-lbl">Bit Indeks</div></div>', unsafe_allow_html=True)

        # Simpan Laporan JSON ke folder sesi
        record = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "Embed",
            "zodiak": zodiak_choice,
            "psnr": round(psnr, 2),
            "frames_embedded": len(temporal_key),
            "video_file": uploaded_video.name,
            "output_video": output_name,
        }
        report_path = os.path.join(session_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=4)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Video Stego (.mp4 lossy)", f, file_name=output_name, mime="video/mp4")


# ============================================================
# HALAMAN: SIMULASI SERANGAN
# ============================================================

def page_attack_simulation():
    st.markdown('<h1 class="main-header">💥 Simulasi Serangan</h1>', unsafe_allow_html=True)
    st.markdown("Laboratorium interaktif untuk mensimulasikan kerusakan video menggunakan serangan geometris maupun pemrosesan sinyal.")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_video = st.file_uploader("Upload Video Stego", type=["mp4", "avi", "mov"], help="Upload video yang telah disisipkan watermark sebelumnya.")
        attack_options = [atk for atk in ATTACK_LABELS if atk != "Clean"]
        attack_type = st.selectbox("Pilih Jenis Serangan", attack_options)

    with col2:
        if uploaded_video:
            st.info(f"📹 File Input: **{uploaded_video.name}** ({uploaded_video.size / 1024:.1f} KB)")
            st.warning(f"Serangan terpilih: **{attack_type}**")

    if st.button("💣 Hancurkan Video!", use_container_width=True, type="primary"):
        if not uploaded_video:
            st.error("Silakan upload video stego terlebih dahulu.")
            return

        session_id = f"Attack_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{attack_type}"
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        with st.spinner(f"Menerapkan serangan {attack_type}..."):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            try:
                frames, fps, w, h = read_all_frames(tmp_path)
                attacked_frames = []
                
                # Progress bar
                progress_bar = st.progress(0)
                total_frames = len(frames)

                for i, frame in enumerate(frames):
                    att_frame = apply_attack(frame, attack_type)
                    attacked_frames.append(att_frame)
                    progress_bar.progress((i + 1) / total_frames)

                output_name = f"attacked_{attack_type}.mp4"
                output_path = os.path.join(session_dir, output_name)
                save_video(attacked_frames, output_path, fps, w, h)

                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Terjadi error saat simulasi serangan: {e}")
                return

        st.success(f"✅ Simulasi berhasil! Video rusak tersimpan di: `{session_dir}`")
        
        # Laporan JSON
        record = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "Attack Simulation",
            "attack_type": attack_type,
            "total_frames": len(attacked_frames),
            "input_video": uploaded_video.name,
            "output_video": output_name,
        }
        report_path = os.path.join(session_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=4)

        with open(output_path, "rb") as f:
            st.download_button("📥 Download Video Hasil Serangan (.mp4)", f, file_name=output_name, mime="video/mp4")

# ============================================================
# HALAMAN: DETEKSI & EKSTRAK
# ============================================================

def page_detect(models: dict):
    st.markdown('<h1 class="main-header">🔍 Deteksi & Ekstrak</h1>', unsafe_allow_html=True)
    st.markdown("Pipeline Triple-AI dengan Temporal Averaging untuk menganalisis video.")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_video = st.file_uploader("Upload Video untuk Dianalisis", type=["mp4", "avi", "mov"])
        password = st.text_input("Kata Sandi Ekstraksi", type="password", placeholder="Masukkan kata sandi yang digunakan saat embed")
    with col2:
        num_frames = st.slider("Jumlah Frame yang Dianalisis", min_value=5, max_value=30, value=10, step=5)
        st.info(f"💡 Semakin banyak frame yang dianalisis, semakin akurat hasil Temporal Averaging-nya.")

    any_model_ready = any(v is not None for v in models.values())
    if not any_model_ready:
        st.warning("⚠️ Model AI belum tersedia. Jalankan dulu script training (Step 2, 3, 4) di terminal.")

    if st.button("🔬 Mulai Analisis", use_container_width=True, type="primary"):
        if not uploaded_video:
            st.error("Silakan upload video terlebih dahulu.")
            return
        if not password:
            st.error("Masukkan kata sandi.")
            return

        # Buat Folder Sesi Unik
        session_id = f"Deteksi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        with st.spinner("Menganalisis video dengan Triple-AI..."):
            # Simpan file yang diupload ke dalam folder sesi agar tersimpan rapi
            video_ext = uploaded_video.name.split(".")[-1]
            saved_video_name = f"uploaded_video.{video_ext}"
            saved_video_path = os.path.join(session_dir, saved_video_name)
            
            with open(saved_video_path, "wb") as f:
                f.write(uploaded_video.read())

            try:
                frames, fps, w, h = read_all_frames(saved_video_path)
                result = run_detection_pipeline(frames, password, models, num_frames)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if "error" in result:
            st.error(result["error"])
            return

        # ===== TAMPILKAN HASIL =====
        st.markdown("---")
        st.markdown("### 🔬 Hasil Investigasi Dual-AI")

        st.markdown('<span class="step-badge">LANGKAH 1</span> **Detektor Keberadaan Watermark (BER Threshold)**', unsafe_allow_html=True)
        ber_val = result["ber_value"]
        if result["has_watermark"]:
            st.markdown(f'<div class="result-box">✅ <b>Terdeteksi Watermark</b> — Bit Error Rate: <b>{ber_val:.2f}</b> (Batas toleransi: 0.30)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ <b>Kerusakan Matematis Tinggi</b> — Bit Error Rate: <b>{ber_val:.2f}</b> (Melewati batas 0.30)</div>', unsafe_allow_html=True)

        st.markdown('<span class="step-badge">LANGKAH 2</span> **Spesialis Analisis Serangan**', unsafe_allow_html=True)
        atk = result["attack_result"]
        atk_conf = result["attack_confidence"]
        if atk == "Clean":
            st.markdown(f'<div class="result-box">🛡️ <b>Video Bersih</b> (Tidak ada serangan) — Kepercayaan: <b>{atk_conf*100:.1f}%</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ <b>Serangan Terdeteksi: {atk}</b> — Kepercayaan: <b>{atk_conf*100:.1f}%</b></div>', unsafe_allow_html=True)

        st.markdown('<span class="step-badge">LANGKAH 3</span> **Spesialis Pengenal Logo Zodiak**', unsafe_allow_html=True)
        logo = result["logo_result"]
        logo_conf = result["logo_confidence"]
        st.markdown(f'<div class="result-box">🔮 <b>Logo Zodiak: {logo}</b> — Kepercayaan Agregat: <b>{logo_conf*100:.1f}%</b></div>', unsafe_allow_html=True)

        img_col, chart_col = st.columns([1, 2])
        with img_col:
            logo_path = os.path.join(LOGOS_DIR, f"{logo}.png")
            if os.path.exists(logo_path):
                st.image(logo_path, caption=f"Identitas Terverifikasi: {logo}", use_container_width=True)
        
        with chart_col:
            if result.get("all_logo_confidence"):
                conf_df = pd.DataFrame(
                    list(result["all_logo_confidence"].items()),
                    columns=["Zodiak", "Kepercayaan"]
                ).sort_values("Kepercayaan", ascending=True)
                st.bar_chart(conf_df.set_index("Zodiak"))

        st.caption(f"📊 Dianalisis dari {result['frames_analyzed']} frame (dari total {result['total_frames']} frame)")
        st.success(f"📂 Hasil dan Laporan otomatis tersimpan di folder: `{session_dir}`")

        # Simpan Laporan JSON ke folder sesi
        record = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "Deteksi",
            "has_watermark": result["has_watermark"],
            "ber_value": round(result["ber_value"], 3),
            "attack_result": result.get("attack_result", "N/A"),
            "attack_confidence": round(result.get("attack_confidence", 0) * 100, 1),
            "logo_result": result.get("logo_result", "N/A"),
            "logo_confidence": round(result.get("logo_confidence", 0) * 100, 1),
            "frames_analyzed": result["frames_analyzed"],
            "video_file": uploaded_video.name,
            "saved_video": saved_video_name,
        }
        report_path = os.path.join(session_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=4)


# ============================================================
# HALAMAN: LAPORAN ML
# ============================================================

def page_reports():
    st.markdown('<h1 class="main-header">📊 Laporan Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown("Hasil evaluasi dari 3 model AI yang telah dilatih.")
    st.divider()

    report_files = {
        "Model 2 - Spesialis Serangan": "report_attack.txt",
        "Model 3 - Spesialis Logo Zodiak": "report_logo.txt",
    }
    cm_files = {
        "Model 2": "confusion_matrix_attack.png",
        "Model 3": "confusion_matrix_logo.png",
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
    st.markdown("### Confusion Matrix")
    cm_cols = st.columns(3)
    for col, (name, cfile) in zip(cm_cols, cm_files.items()):
        with col:
            cpath = os.path.join(REPORTS_DIR, cfile)
            if os.path.exists(cpath):
                st.image(cpath, caption=name, use_container_width=True)
            else:
                st.info(f"{name}: Belum ada grafik.")


# ============================================================
# HALAMAN: FOLDER OUTPUT SESI
# ============================================================

def page_history():
    st.markdown('<h1 class="main-header">📂 Folder Output Sesi</h1>', unsafe_allow_html=True)
    st.markdown("Semua riwayat Embed dan Deteksi tersimpan rapi dalam folder unik.")
    st.divider()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Membaca seluruh sub-folder di dalam OUTPUT_DIR
    session_folders = [f for f in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
    session_folders.sort(reverse=True)  # Urutkan terbaru (karena format namanya Timestamp)

    if not session_folders:
        st.info("Belum ada sesi tersimpan. Lakukan operasi Embed atau Deteksi untuk membuat folder Sesi baru.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_session = st.selectbox("📁 Pilih Folder Sesi:", session_folders)
        session_path = os.path.join(OUTPUT_DIR, selected_session)
        
        st.markdown("### Isi Folder:")
        files_in_folder = os.listdir(session_path)
        for f in files_in_folder:
            st.markdown(f"- 📄 `{f}`")

    with col2:
        st.markdown(f"### 📋 Rincian: `{selected_session}`")
        report_file = os.path.join(session_path, "report.json")
        
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                record = json.load(f)
            
            st.json(record)
            
            # Cari file video di dalam folder (.avi atau .mp4)
            video_files = [f for f in files_in_folder if f.endswith(('.avi', '.mp4'))]
            if video_files:
                for vf in video_files:
                    vpath = os.path.join(session_path, vf)
                    st.success(f"🎬 Terdapat file video: `{vf}`")
                    with open(vpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {vf}",
                            data=f,
                            file_name=vf,
                            mime="video/x-msvideo" if vf.endswith('.avi') else "video/mp4"
                        )
        else:
            st.warning("File report.json tidak ditemukan di dalam folder ini.")


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
