# =============================================================================
# ml_pipeline/2_train_master_model.py
# Training Model Master: Arsitektur Hibrida (4-Bit Serangan + 1 Skalar Logo)
# Input  : X0 (11) + Xw (11) + Xa (11) = 33 Fitur
# Target : 5 Kolom [y_atk_bit1..4 (biner) | y_logo_scalar (desimal 0-7)]
# VERSI KOMPARASI: Membandingkan 4 Algoritma ML, menyimpan yang terbaik
# =============================================================================

import os
import sys
import pickle
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    DATASET_MASTER_PATH, MODEL_MASTER_PATH, REPORTS_DIR,
    MASTER_FEATURE_COLUMNS, MASTER_TARGET_COLUMNS,
    ML_TEST_SIZE, ML_RANDOM_STATE,
    ZODIAK_LABELS, ATTACK_LABELS
)
from core.security import decode_hybrid_labels


# ---- Definisi 4 Kandidat Algoritma (Base Models) ----
# Dibungkus MultiOutputClassifier agar bisa nebak 7 kolom sekaligus
CANDIDATE_MODELS = {
    "Random Forest": MultiOutputClassifier(
        RandomForestClassifier(random_state=ML_RANDOM_STATE, n_estimators=100, class_weight="balanced"),
        n_jobs=-1
    ),
    "Decision Tree": MultiOutputClassifier(
        DecisionTreeClassifier(random_state=ML_RANDOM_STATE, class_weight="balanced"),
        n_jobs=-1
    ),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MultiOutputClassifier(
            SVC(kernel="rbf", probability=False, class_weight="balanced",
                random_state=ML_RANDOM_STATE),
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": MultiOutputClassifier(
        GradientBoostingClassifier(random_state=ML_RANDOM_STATE, n_estimators=100),
        n_jobs=-1
    ),
}


def compute_breakdown_accuracy(model, X_test, y_test):
    """
    Menghitung 3 metrik akurasi dari prediksi hibrida (5 kolom):
    1. Akurasi Serangan: Apakah 4 bit pertama SEMUA benar?
    2. Akurasi Logo    : Apakah skalar logo (kolom ke-5) tepat?
    3. Akurasi Overall : Apakah semua 5 kolom SEMUA benar? (Paling ketat)

    Returns: (overall_acc, attack_acc, zodiak_acc, decoded_labels_pred, decoded_labels_true)
    """
    y_pred = model.predict(X_test)  # shape: (n_samples, 5)

    n = len(y_test)

    # Akurasi serangan: 4 bit pertama (index 0-3)
    attack_correct = np.all(y_pred[:, :4] == y_test[:, :4], axis=1)
    attack_acc = float(np.sum(attack_correct) / n)

    # Akurasi logo: skalar tunggal di kolom terakhir (index 4)
    logo_correct = (y_pred[:, 4] == y_test[:, 4])
    logo_acc = float(np.sum(logo_correct) / n)

    # Akurasi overall: semua 5 kolom harus benar
    overall_correct = np.all(y_pred == y_test, axis=1)
    overall_acc = float(np.sum(overall_correct) / n)

    # Decode tebakan ke label teks (pakai decoder hibrida)
    decoded_pred = [decode_hybrid_labels(row) for row in y_pred]
    decoded_true = [decode_hybrid_labels(row) for row in y_test]

    return overall_acc, attack_acc, logo_acc, decoded_pred, decoded_true, y_pred


def train_master_model():
    start_time = time.time()

    print("=" * 70)
    print("TRAINING MODEL MASTER - ARSITEKTUR HIBRIDA")
    print("Input  : X0 + Xw + Xa = 33 Fitur")
    print("Target : 5 Kolom [4 bit Serangan (biner) | 1 Skalar Logo (desimal)]")
    print("MODE   : STUDI KOMPARASI 4 ALGORITMA")
    print("=" * 70)

    if not os.path.exists(DATASET_MASTER_PATH):
        print(f"[ERROR] Dataset tidak ditemukan: {DATASET_MASTER_PATH}")
        print("        Jalankan dulu: python ml_pipeline/1_generate_dataset.py")
        return

    df = pd.read_csv(DATASET_MASTER_PATH)
    print(f"Dataset master dimuat: {len(df)} baris, {len(df.columns)} kolom")

    # Validasi kolom
    all_required = MASTER_FEATURE_COLUMNS + MASTER_TARGET_COLUMNS
    missing_cols = [c for c in all_required if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Kolom tidak ditemukan: {missing_cols[:5]}...")
        return

    # Bersihkan NaN
    X = df[MASTER_FEATURE_COLUMNS].values.astype(np.float64)
    Y = df[MASTER_TARGET_COLUMNS].values.astype(int)
    mask = ~np.isnan(X).any(axis=1)
    X, Y = X[mask], Y[mask]
    print(f"Data setelah pembersihan NaN: {len(X)} baris")
    print(f"Ukuran X: {X.shape} | Ukuran Y: {Y.shape}")

    # Train-Test Split (Stratify pakai kolom pertama Y sebagai proxy)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ============================
    # KOMPARASI 4 ALGORITMA
    # ============================
    results = {}
    print(f"{'-'*70}")
    print(f"{'Algoritma':<22} {'Overall':>10} {'Serangan':>10} {'Logo':>10} {'Waktu':>8}")
    print(f"{'-'*70}")

    for name, model in CANDIDATE_MODELS.items():
        t0 = time.time()
        print(f"\n[INFO] Training {name}...")

        model.fit(X_train, Y_train)

        overall_acc, attack_acc, logo_acc, dec_pred, dec_true, y_pred_raw = \
            compute_breakdown_accuracy(model, X_test, Y_test)

        # Hitung Hamming Loss secara manual (fraction of incorrect labels)
        # sklearn.metrics.hamming_loss tidak support multiclass-multioutput (Hibrida)
        h_loss = float(np.mean(Y_test != y_pred_raw))
        elapsed = time.time() - t0

        results[name] = {
            "model": model,
            "overall_acc": overall_acc,
            "attack_acc": attack_acc,
            "logo_acc": logo_acc,
            "hamming_loss": h_loss,
            "decoded_pred": dec_pred,
            "decoded_true": dec_true,
            "time": elapsed,
        }
        print(f"  {name:<20} {overall_acc*100:>9.2f}%  {attack_acc*100:>9.2f}%  "
              f"{logo_acc*100:>9.2f}%  {elapsed:>6.1f}s  (Hamming Loss: {h_loss:.4f})")

    print(f"\n{'-'*70}")

    # Pilih pemenang berdasarkan akurasi Logo tertinggi
    best_name = max(results, key=lambda n: results[n]["logo_acc"])
    best = results[best_name]

    print(f"\n🏆 PEMENANG: {best_name}")
    print(f"   Akurasi Overall (5 kolom tepat semua) : {best['overall_acc']*100:.2f}%")
    print(f"   Akurasi Serangan (4 bit tepat)        : {best['attack_acc']*100:.2f}%")
    print(f"   Akurasi Logo Zodiak (skalar tepat)    : {best['logo_acc']*100:.2f}%")
    print(f"   Hamming Loss                          : {best['hamming_loss']:.4f}")

    # ============================
    # EVALUASI MENDALAM UNTUK PEMENANG
    # ============================
    print(f"\n[INFO] Menjalankan Evaluasi Mendalam (CV, F1, Confusion Matrix) untuk {best_name}...")
    
    # 1. K-Fold Cross Validation (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=ML_RANDOM_STATE)
    cv_overall_accs = []
    
    # Kita butuh model baru yang fresh untuk CV
    cv_model = CANDIDATE_MODELS[best_name]
    
    print("       -> Menjalankan 5-Fold Cross Validation...")
    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        cv_model.fit(X_tr, Y_tr)
        y_val_pred = cv_model.predict(X_val)
        # Hitung akurasi overall untuk fold ini
        fold_acc = float(np.sum(np.all(y_val_pred == Y_val, axis=1)) / len(Y_val))
        cv_overall_accs.append(fold_acc)
        fold_idx += 1
    
    cv_mean = np.mean(cv_overall_accs) * 100
    cv_std = np.std(cv_overall_accs) * 100
    print(f"       -> Hasil 5-Fold CV (Overall Acc): {cv_mean:.2f}% ± {cv_std:.2f}%")

    # ---- Laporan Teks Mendalam & Confusion Matrix PER MODEL ----
    # Buat string penampung untuk laporan teks mendalam semua model
    detailed_reports = []
    
    for model_name, res in results.items():
        # Hitung Ulang Classification Report & Confusion Matrix
        true_atks = [dec[0] for dec in res["decoded_true"]]
        pred_atks = [dec[0] for dec in res["decoded_pred"]]
        true_logos = [dec[1] for dec in res["decoded_true"]]
        pred_logos = [dec[1] for dec in res["decoded_pred"]]

        report_atk = classification_report(true_atks, pred_atks, zero_division=0)
        report_logo = classification_report(true_logos, pred_logos, zero_division=0)
        
        cm_atk = confusion_matrix(true_atks, pred_atks, labels=ATTACK_LABELS)
        cm_logo = confusion_matrix(true_logos, pred_logos, labels=ZODIAK_LABELS)
        
        # Tambahkan teks ke daftar laporan mendalam
        detailed_text = f"\n{'=' * 70}\n"
        detailed_text += f"EVALUASI MENDALAM: {model_name}\n"
        detailed_text += f"{'=' * 70}\n"
        detailed_text += "--- CLASSIFICATION REPORT: SERANGAN ---\n"
        detailed_text += report_atk + "\n"
        detailed_text += "--- CLASSIFICATION REPORT: LOGO ZODIAK ---\n"
        detailed_text += report_logo + "\n"
        detailed_reports.append(detailed_text)
        
        # Simpan Gambar Confusion Matrix Serangan
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        atk_cm_path = os.path.join(REPORTS_DIR, f"{safe_name}_confusion_matrix_attack.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_atk, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=ATTACK_LABELS, yticklabels=ATTACK_LABELS, ax=ax)
        ax.set_title(f"Confusion Matrix Serangan ({model_name})", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(atk_cm_path, dpi=150)
        plt.close()
        
        # Simpan Gambar Confusion Matrix Logo Zodiak
        logo_cm_path = os.path.join(REPORTS_DIR, f"{safe_name}_confusion_matrix_logo.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_logo, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=ZODIAK_LABELS, yticklabels=ZODIAK_LABELS, ax=ax)
        ax.set_title(f"Confusion Matrix Logo Zodiak ({model_name})", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(logo_cm_path, dpi=150)

    # Simpan model pemenang ke disk
    print(f"\n[INFO] Menyimpan model pemenang ke {MODEL_MASTER_PATH}...")
    master_bundle = {
        "models": {best_name: best["model"]},
        "best_algorithm": best_name,
        "feature_columns": MASTER_FEATURE_COLUMNS,
        "target_columns": MASTER_TARGET_COLUMNS,
        "zodiak_labels": ZODIAK_LABELS,
        "attack_labels": ATTACK_LABELS
    }
    with open(MODEL_MASTER_PATH, "wb") as f:
        pickle.dump(master_bundle, f)
    print(f"[SAVED] Model berhasil disimpan!")

    # Tulis laporan ke dalam satu file .txt
    report_path = os.path.join(REPORTS_DIR, "report_master.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LAPORAN EVALUASI - MODEL MASTER (ARSITEKTUR HIBRIDA)\n")
        f.write("STUDI KOMPARASI 4 ALGORITMA - 4-BIT SERANGAN + 1 SKALAR LOGO\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Algoritma':<22} {'Overall':>10} {'Serangan':>10} {'Logo':>10} {'HammingLoss':>13} {'Waktu':>8}\n")
        f.write(f"{'-'*70}\n")
        for name, res in results.items():
            marker = " <- TERBAIK" if name == best_name else ""
            f.write(
                f"  {name:<20} {res['overall_acc']*100:>9.2f}%  {res['attack_acc']*100:>9.2f}%  "
                f"{res['logo_acc']*100:>9.2f}%  {res['hamming_loss']:>12.4f}  {res['time']:>6.1f}s{marker}\n"
            )
        f.write(f"\nModel Terpilih  : {best_name}\n")
        f.write(f"Akurasi Overall : {best['overall_acc']*100:.2f}%\n")
        f.write(f"Akurasi Serangan: {best['attack_acc']*100:.2f}%\n")
        f.write(f"Akurasi Logo    : {best['logo_acc']*100:.2f}%\n")
        f.write(f"Hamming Loss    : {best['hamming_loss']:.4f}\n\n")
        
        # Tambahkan cross validation model pemenang
        f.write("=" * 70 + "\n")
        f.write(f"5-Fold Cross Validation (Pemenang: {best_name})\n")
        f.write("=" * 70 + "\n")
        f.write(f"Mean : {cv_mean:.2f}%\n")
        f.write(f"Std  : {cv_std:.2f}%\n")
        
        # Masukkan semua detil classification report
        for detailed_text in detailed_reports:
            f.write(detailed_text)
            
        f.write(f"\nTotal Waktu Seluruh Proses: {time.time()-start_time:.2f} detik\n")
    print(f"[SAVED] Laporan mendalam 4 model disimpan ke: {report_path}")
    print(f"[SAVED] 8 Grafik Confusion Matrix (tiap model) disimpan ke folder reports/")

    # ---- Grafik 1: Komparasi 3 Metrik Akurasi per Algoritma ----
    comp_path = os.path.join(REPORTS_DIR, "comparison_master.png")
    names = list(results.keys())
    overall_accs = [results[n]["overall_acc"] * 100 for n in names]
    attack_accs  = [results[n]["attack_acc"] * 100 for n in names]
    logo_accs    = [results[n]["logo_acc"] * 100 for n in names]
    colors_win   = ["#F59E0B" if n == best_name else "#6366F1" for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    w = 0.28
    bars1 = ax.bar(x - w, overall_accs, w, label="Akurasi Overall (5 kolom)", color=colors_win, alpha=0.9)
    bars2 = ax.bar(x,     attack_accs,  w, label="Akurasi Serangan (4 bit)", color="#EC4899", alpha=0.75)
    bars3 = ax.bar(x + w, logo_accs,    w, label="Akurasi Logo Zodiak (skalar)", color="#10B981", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Akurasi (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Komparasi 4 Algoritma - Model Master Hibrida (4-Bit + Skalar)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"[SAVED] Grafik komparasi (Bar Chart) disimpan ke: {comp_path}")

    print(f"\n[OK] Selesai dalam {time.time()-start_time:.2f} detik!")
    print("     Semua model siap! Jalankan: streamlit run app/dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    train_master_model()
