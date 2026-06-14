# =============================================================================
# ml_pipeline/4_train_logo_specialist.py
# Training Model 3: Spesialis Pengenal Logo Zodiak (8 Kelas, Input 64-bit)
# VERSI KOMPARASI: Membandingkan 4 Algoritma ML sekaligus, menyimpan yang terbaik
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    DATASET_LOGO_PATH, MODEL_LOGO_PATH, REPORTS_DIR,
    LOGO_FEATURE_COLUMNS, ZODIAK_LABELS, ML_TEST_SIZE,
    ML_RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH
)

# ---- Definisi 4 Kandidat Algoritma (Base Models) ----
CANDIDATE_MODELS = {
    "Random Forest": RandomForestClassifier(random_state=ML_RANDOM_STATE, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=ML_RANDOM_STATE, class_weight="balanced"),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=ML_RANDOM_STATE))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(random_state=ML_RANDOM_STATE),
}

# ---- Definisi Ruang Pencarian Hyperparameter (Grid Search) ----
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    },
    "Decision Tree": {
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5],
    },
    "SVM (RBF)": {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.1, 0.01],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
}


def train_logo_specialist():
    start_time = time.time()

    print("=" * 70)
    print("TRAINING MODEL 3: SPESIALIS PENGENAL LOGO ZODIAK")
    print("MODE: STUDI KOMPARASI 4 ALGORITMA")
    print("=" * 70)

    if not os.path.exists(DATASET_LOGO_PATH):
        print(f"[ERROR] Dataset tidak ditemukan: {DATASET_LOGO_PATH}")
        return

    df = pd.read_csv(DATASET_LOGO_PATH)
    print(f"Dataset logo dimuat: {len(df)} baris")
    print(f"Distribusi kelas zodiak:\n{df['label_zodiak'].value_counts().to_string()}")

    # Filter baris yang mengandung NaN secara menyeluruh
    X_temp = df[LOGO_FEATURE_COLUMNS].values.astype(np.float64)
    mask = ~np.isnan(X_temp).any(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    
    X = df_clean[LOGO_FEATURE_COLUMNS].values.astype(np.float64)
    y = df_clean["label_zodiak"].values
    
    print(f"\nData setelah pembersihan NaN: {len(X)} baris")
    print(f"Ukuran vektor fitur: {X.shape[1]} bit (64-bit PN Sequence)")

    # Lakukan split, dan simpan indexnya agar kita bisa memfilter df_clean nanti
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )
    
    # Ambil subset df_clean yang benar-benar merupakan Test Set
    df_test = df_clean.iloc[idx_test].copy()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    class_names = sorted(df["label_zodiak"].unique().tolist())

    # ============================
    # KOMPARASI 4 ALGORITMA
    # ============================
    results = {}
    print(f"\n{'─'*70}")
    print(f"{'Algoritma':<22} {'Akurasi Test':>13} {'CV Mean':>10} {'CV Std':>8} {'Waktu':>8}")
    print(f"{'─'*70}")

    for name, base_model in CANDIDATE_MODELS.items():
        t0 = time.time()
        print(f"\n[INFO] Menjalankan GridSearchCV untuk {name}...")
        
        # Inisialisasi GridSearchCV (5-Fold CV)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRIDS[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        
        # Training & Pencarian Parameter
        grid_search.fit(X_train, y_train)
        
        # Evaluasi menggunakan model terbaik
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        cv_mean = grid_search.best_score_
        cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        
        elapsed = time.time() - t0

        results[name] = {
            "model": best_model,
            "accuracy": acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "y_pred": y_pred,
            "best_params": grid_search.best_params_,
            "time": elapsed,
        }
        print(f"  {name:<20} {acc*100:>12.2f}%  {cv_mean*100:>9.2f}%  {cv_std*100:>6.2f}%  {elapsed:>6.1f}s")
        print(f"  Best Params: {grid_search.best_params_}")

    print(f"{'─'*70}")

    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best = results[best_name]
    print(f"\n🏆 PEMENANG: {best_name}")
    print(f"   Akurasi Test : {best['accuracy']*100:.2f}%")
    print(f"   CV (5-fold)  : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%")

    report = classification_report(y_test, best["y_pred"], target_names=class_names)
    cm = confusion_matrix(y_test, best["y_pred"], labels=class_names)
    print(f"\nClassification Report ({best_name}):\n{report}")

    # ---- Akurasi per Serangan ----
    attack_acc_rows = []
    if "attack_applied" in df_test.columns:
        print("Akurasi per Kondisi Serangan (hanya pada Test Set):")
        for atk in sorted(df_test["attack_applied"].unique()):
            mask_atk = df_test["attack_applied"] == atk
            X_atk = df_test[mask_atk][LOGO_FEATURE_COLUMNS].values.astype(np.float64)
            y_atk = df_test[mask_atk]["label_zodiak"].values
            if len(X_atk) > 0:
                y_pred_atk = best["model"].predict(X_atk)
                atk_acc = accuracy_score(y_atk, y_pred_atk)
                print(f"  {atk:22s}: {atk_acc*100:.2f}%")
                attack_acc_rows.append((atk, atk_acc))

    with open(MODEL_LOGO_PATH, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"\n[SAVED] Model terbaik ({best_name}) disimpan ke: {MODEL_LOGO_PATH}")

    # ---- Simpan Laporan Teks ----
    report_path = os.path.join(REPORTS_DIR, "report_logo.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LAPORAN EVALUASI - MODEL 3: SPESIALIS LOGO ZODIAK\n")
        f.write("STUDI KOMPARASI 4 ALGORITMA\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Algoritma':<22} {'Akurasi Test':>13} {'CV Mean':>10} {'CV Std':>8} {'Waktu':>8}\n")
        f.write(f"{'─'*70}\n")
        for name, res in results.items():
            marker = " ← TERBAIK" if name == best_name else ""
            f.write(f"  {name:<20} {res['accuracy']*100:>12.2f}%  {res['cv_mean']*100:>9.2f}%  {res['cv_std']*100:>6.2f}%  {res['time']:>6.1f}s{marker}\n")
            f.write(f"  Params: {res['best_params']}\n\n")
        f.write(f"\nModel Terpilih : {best_name}\n")
        f.write(f"Best Params    : {best['best_params']}\n")
        f.write(f"Akurasi Test   : {best['accuracy']*100:.2f}%\n")
        f.write(f"CV (5-fold)    : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%\n\n")
        f.write(f"Kelas Zodiak: {class_names}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        if attack_acc_rows:
            f.write("Akurasi per Serangan:\n")
            for atk, atk_acc in attack_acc_rows:
                f.write(f"  {atk:22s}: {atk_acc*100:.2f}%\n")
        f.write(f"\nTotal Waktu: {time.time()-start_time:.2f} detik\n")
    print(f"[SAVED] Laporan disimpan ke: {report_path}")

    # ---- Grafik 1: Komparasi Akurasi ----
    comp_path = os.path.join(REPORTS_DIR, "comparison_logo.png")
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    cvs  = [results[n]["cv_mean"] * 100 for n in names]
    colors = ["#F59E0B" if n == best_name else "#10B981" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x - 0.2, accs, 0.35, label="Akurasi Test Set", color=colors, alpha=0.9)
    ax.bar(x + 0.2, cvs, 0.35, label="CV Mean (5-fold)", color=colors, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Akurasi (%)"); ax.set_ylim(0, 110)
    ax.set_title("Komparasi Algoritma - Model 3: Spesialis Logo Zodiak", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150); plt.close()
    print(f"[SAVED] Grafik komparasi: {comp_path}")

    # ---- Grafik 2: Akurasi per Serangan ----
    if attack_acc_rows:
        atk_path = os.path.join(REPORTS_DIR, "logo_accuracy_per_attack.png")
        atk_names = [r[0] for r in attack_acc_rows]
        atk_accs  = [r[1]*100 for r in attack_acc_rows]
        colors_atk = ["#22C55E" if a >= 80 else "#F97316" if a >= 60 else "#EF4444" for a in atk_accs]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(atk_names, atk_accs, color=colors_atk, alpha=0.9)
        ax.set_ylabel("Akurasi (%)"); ax.set_ylim(0, 110)
        ax.set_title(f"Akurasi Logo Zodiak per Kondisi Serangan ({best_name})", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(atk_path, dpi=150); plt.close()
        print(f"[SAVED] Grafik per serangan: {atk_path}")

    # ---- Grafik 3: Confusion Matrix Pemenang ----
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix_logo.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}\nAkurasi: {best['accuracy']*100:.2f}%", fontsize=14)
    ax.set_xlabel("Prediksi", fontsize=12); ax.set_ylabel("Aktual", fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(cm_path, dpi=150); plt.close()
    print(f"[SAVED] Confusion matrix: {cm_path}")

    print(f"\n[OK] Selesai dalam {time.time()-start_time:.2f} detik!")
    print("     Semua model siap! Jalankan: streamlit run app/dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    train_logo_specialist()
