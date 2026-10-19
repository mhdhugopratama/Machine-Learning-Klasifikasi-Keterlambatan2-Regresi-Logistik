from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_ekspedisi.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Baca Dataset
df = pd.read_csv(DATA_PATH)

# 2. Transformasi Fitur (Label Encoding)
LAYANAN_MAP = {"Ekspres": 0, "Kargo": 1, "Reguler": 2, "Sameday": 3}
df["X1"] = df["Jarak_Pengiriman_KM"]
df["X2"] = df["Jumlah_Transit"]
df["X3"] = df["Jenis_Layanan"].map(LAYANAN_MAP)
df["Y"] = (df["Status_Keterlambatan"] == "Terlambat").astype(int)

# 3. Parameter Model & Ambang Batas Sensitif (Threshold 0.25)
B0, B1, B2, B3 = -1.2025, -0.0001, -0.0054, 0.1300
THRESHOLD = 0.25

def predict_probabilities(data_df, th=THRESHOLD):
    z = B0 + (B1 * data_df["X1"]) + (B2 * data_df["X2"]) + (B3 * data_df["X3"])
    p = 1.0 / (1.0 + np.exp(-z))
    pred = (p >= th).astype(int)
    return z, p, pred

# 4. Evaluasi Seluruh Dataset (500 Data)
z_all, p_all, pred_all = predict_probabilities(df)
y_all = df["Y"].values
cm_all = confusion_matrix(y_all, pred_all, labels=[1, 0])

eval_all = {
    "n_samples": int(len(df)),
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(y_all, pred_all)),
    "precision": float(precision_score(y_all, pred_all, zero_division=0)),
    "recall": float(recall_score(y_all, pred_all, zero_division=0)),
    "f1_score": float(f1_score(y_all, pred_all, zero_division=0)),
    "confusion_matrix": {
        "TP": int(cm_all[0, 0]),
        "FN": int(cm_all[0, 1]),
        "FP": int(cm_all[1, 0]),
        "TN": int(cm_all[1, 1])
    },
    "prob_max": float(p_all.max()),
    "prob_min": float(p_all.min()),
    "prob_mean": float(p_all.mean())
}

# 5. Evaluasi Data Uji (Split 80:20 = 100 Data)
X_train, X_test, y_train, y_test = train_test_split(df, df["Y"], test_size=0.2, random_state=42, stratify=df["Y"])
z_test, p_test, pred_test = predict_probabilities(X_test)
cm_test = confusion_matrix(y_test, pred_test, labels=[1, 0])

eval_test = {
    "n_samples": int(len(X_test)),
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(y_test, pred_test)),
    "precision": float(precision_score(y_test, pred_test, zero_division=0)),
    "recall": float(recall_score(y_test, pred_test, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred_test, zero_division=0)),
    "confusion_matrix": {
        "TP": int(cm_test[0, 0]),
        "FN": int(cm_test[0, 1]),
        "FP": int(cm_test[1, 0]),
        "TN": int(cm_test[1, 1])
    }
}

# 6. Simpan Model & Metadata
model = LogisticRegression()
model.classes_ = np.array([0, 1])
model.intercept_ = np.array([B0])
model.coef_ = np.array([[B1, B2, B3]])
model.feature_names_in_ = np.array(["Jarak_Pengiriman_KM", "Jumlah_Transit", "Jenis_Layanan_Encoded"])

payload = {
    "parameters": {"beta_0": B0, "beta_1": B1, "beta_2": B2, "beta_3": B3},
    "layanan_encoding": LAYANAN_MAP,
    "threshold": THRESHOLD,
    "evaluasi_seluruh_data": eval_all,
    "evaluasi_data_uji": eval_test
}

joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")
with open(MODEL_DIR / "evaluation.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("=" * 65)
print(f"EVALUASI MODEL DENGAN AMBANG BATAS {THRESHOLD} (SENSITIF TERLAMBAT)")
print("=" * 65)
print(f"1. Seluruh Dataset (500 baris):")
print(f"   Akurasi : {eval_all['accuracy']:.2%} | Presisi : {eval_all['precision']:.2%} | Recall : {eval_all['recall']:.2%}")
print(f"   TP (Terlambat Terdeteksi): {eval_all['confusion_matrix']['TP']} dari 127 paket terlambat!")
print(f"   TN: {eval_all['confusion_matrix']['TN']} | FP: {eval_all['confusion_matrix']['FP']} | FN: {eval_all['confusion_matrix']['FN']}")
print(f"2. Data Uji Test Split (100 baris):")
print(f"   Akurasi : {eval_test['accuracy']:.2%} | Recall : {eval_test['recall']:.2%}")
print(f"   TP: {eval_test['confusion_matrix']['TP']} dari 25 paket terlambat")
print("=" * 65)
