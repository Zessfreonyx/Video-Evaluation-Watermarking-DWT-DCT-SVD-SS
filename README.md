# ♈ Zodiac Watermark Steganography: Hybrid DWT-DCT-SVD-SS + Triple-AI Pipeline

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

Sebuah framework *Testkit Digital Video Steganography & Watermarking* skala *Enterprise*. Proyek ini menyembunyikan payload rahasia 64-bit (Sandi Zodiak) ke dalam video MP4/AVI menggunakan algoritma hibrida tingkat lanjut **(DWT + DCT + SVD)** yang dikombinasikan dengan metode **Spread Spectrum (SS)**.

Untuk bertahan dari serangan pemrosesan sinyal dan kompresi yang sangat brutal, sistem ekstraksi (pembaca) diperkuat menggunakan arsitektur **Triple-AI Machine Learning** dan *Temporal Averaging*. Hal ini memungkinkan AI untuk membaca pesan rahasia yang hancur dari dalam video yang telah rusak parah (Misal: Kompresi JPEG 50%, Pengecilan Resolusi 50%, dan Gaussian Noise).

## 🌟 Fitur Utama

- **Hybrid Frequency Transform:** Menggabungkan *Discrete Wavelet Transform* (DWT), *Discrete Cosine Transform* (DCT), dan *Singular Value Decomposition* (SVD) untuk mencapai keseimbangan antara *Invisibility* (Tidak terlihat mata) dan *Robustness* (Ketahanan).
- **Spread Spectrum Encoding:** Menyebar pesan rahasia ke seluruh spektrum frekuensi video agar bisa bertahan dari pemotongan data saat kompresi *lossy* MP4.
- **Arsitektur Triple-AI:**
  - **AI Model 1 (Detector):** Analisis Matematika *Bit Error Rate* (BER) Thresholding.
  - **AI Model 2 (Attack Specialist):** Mengklasifikasikan jenis serangan apa yang baru saja melukai video (Algoritma: *Random Forest*).
  - **AI Model 3 (Payload Specialist):** Merekonstruksi data 64-bit yang hancur untuk menebak logo Zodiak asli (Algoritma: *Gradient Boosting*).
- **Interactive UI Testkit:** Dashboard antarmuka *Streamlit* yang sangat ramah pengguna untuk operasi dari hulu ke hilir (*Embed*, *Attack Simulation*, dan *Detect*).
- **Automated Dataset Generation:** *Pipeline* otomatis untuk memproduksi ribuan dataset dan melatih *Machine Learning* secara mandiri.

## 🚀 Cara Menjalankan Project

### 1. Persiapan Installasi
*Clone repository* ini dan *install* seluruh *dependencies*.
```bash
git clone https://github.com/Zessfreonyx/Video-Evaluation-Watermarking-DWT-DCT-SVD-SS.git
cd Video-Evaluation-Watermarking-DWT-DCT-SVD-SS
pip install -r requirements.txt
```

### 2. Siapkan Video Umpan (Host Data)
Masukkan beberapa video MP4/AVI pendek dan bersih (sekitar 3-5 detik) ke dalam folder `data/raw/`. Video-video ini akan digunakan oleh mesin untuk memproduksi ribuan data latih AI.

### 3. Jalankan Mesin Pembelajaran (Training Pipeline)
Jalankan deretan skrip di bawah ini untuk melatih AI di komputer Anda sendiri. Skrip ini akan menyiksa video Anda dengan serangan (Noise, Blur, JPEG, Crop) secara otomatis.
```bash
python ml_pipeline/1_generate_dataset.py
python ml_pipeline/3_train_attack_specialist.py
python ml_pipeline/4_train_logo_specialist.py
```

### 4. Luncurkan Dashboard Testkit
Setelah semua AI lulus pelatihan (muncul di folder `models/`), luncurkan Dashboard *Streamlit*.
```bash
streamlit run app/dashboard.py
```

## 🧪 Laboratorium Simulasi Serangan (Attack Lab)
Dashboard ini dilengkapi dengan *Attack Lab* interaktif. Anda bisa mengunggah video Stego dan menyiksanya dengan kondisi paling ekstrim:
- **JPEG Compression (Quality = 50)**
- **Resize (Scale 50%)**
- **Gaussian Noise (STD = 15)**
- **Cropping & Rotation**

## ⚙️ Parameter Konfigurasi Utama
Sistem ini menggunakan parameter yang dikalibrasi secara khusus untuk menahan serangan tingkat ekstrim. Pengaturan ini digembok di dalam file `config.py`:
- **SVD Scaling Factor (Alpha) = 250.0:** Menentukan tingkat ketebalan/kekuatan *watermark* yang ditanamkan pada nilai singular matriks SVD. Nilai 250 adalah titik keseimbangan (Trade-Off) optimal di mana *watermark* sangat sulit dihancurkan oleh kompresi tanpa membuat video menjadi rusak parah secara visual.
- **Spread Spectrum Repetition = 15:** Menyebarkan dan mengulangi setiap 1-bit data rahasia ke dalam 15 lokasi frekuensi yang acak. Berfungsi sebagai perisai pelindung agar data tetap bisa dibaca meskipun beberapa blok video terpotong (*Cropping*) atau terkena *noise*.

## ⚠️ Known Limitations (Batasan Sistem)
Konsep matematis pemetaan grid spasial (Blok DWT 8x8) sangat rentan terhadap **Video Transcoders** (seperti mengirim video via WhatsApp/TikTok). Platform tersebut merubah secara permanen struktur resolusi geometris dan *framerate* video (*Desynchronization Attack*). Arsitektur *Machine Learning* di sistem ini dibuat untuk menahan gempuran kerusakan sinyal visual, namun transformasi dimensi total akan mengakibatkan data 64-bit hancur sepenuhnya.

## 🎓 Konteks Akademik
Repositori *Open-Source* ini dikembangkan sebagai bagian dari Tugas Akhir (Skripsi) Universitas Telkom yang berfokus pada Pengolahan Citra Digital Tingkat Lanjut (*Advanced Digital Image Processing*) dan Ketahanan *Artificial Intelligence*.

---
*Dikembangkan oleh **Zessfreonyx** - **Universitas Telkom***
