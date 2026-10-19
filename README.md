# Klasifikasi Biner Keterlambatan Pengiriman — Regresi Logistik

Proyek ini mengimplementasikan algoritma **Regresi Logistik** untuk mengklasifikasikan status keterlambatan pengiriman paket ekspedisi ke dalam dua kategori biner: **Terlambat ($Y = 1$)** dan **Tepat Waktu ($Y = 0$)** menggunakan dataset ekspedisi (500 baris data).

---

## 1. Variabel dan Transformasi Data
- **$X_1$**: `Jarak_Pengiriman_KM` (Jarak tempuh pengiriman dalam KM)
- **$X_2$**: `Jumlah_Transit` (Jumlah titik perpindahan hub paket, 0–4 kali)
- **$X_3$**: `Jenis_Layanan` dengan **Label Encoding**:
  - `Ekspres` = `0`
  - `Kargo` = `1`
  - `Reguler` = `2`
  - `Sameday` = `3`
- **$Y$ (Target)**: `Status_Keterlambatan` (`0 = Tepat Waktu`, `1 = Terlambat`)

---

## 2. Persamaan Logit & Fungsi Sigmoid
Parameter diestimasi menggunakan metode *Maximum Likelihood Estimation* (MLE) dengan pembulatan 4 desimal:

$$z = -1.2025 + (-0.0001 \times X_1) + (-0.0054 \times X_2) + (0.1300 \times X_3)$$

Fungsi Sigmoid:
$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-z}}$$

Aturan Keputusan:
- Jika $P(Y = 1 \mid X) \ge 0.5 \implies$ **Terlambat (1)**
- Jika $P(Y = 1 \mid X) < 0.5 \implies$ **Tepat Waktu (0)**

---

## 3. Hasil Evaluasi Model

### A. Seluruh Dataset (500 Baris Data)
- **Akurasi**: 74.60%
- **Confusion Matrix**:
  - True Negative (TN) = 373
  - False Negative (FN) = 127
  - True Positive (TP) = 0
  - False Positive (FP) = 0

### B. Data Uji (Test Split 20% = 100 Baris Data)
- **Akurasi**: 75.00%
- **Confusion Matrix**:
  - True Negative (TN) = 75
  - False Negative (FN) = 25
  - True Positive (TP) = 0
  - False Positive (FP) = 0

---

## 4. Cara Menjalankan di Windows

Buka terminal PowerShell di folder proyek ini:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
