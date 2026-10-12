# Rice-Cammeo-and-Osmancik-Classification

Project ini merupakan praktikum Machine Learning yang menggunakan dataset Rice Cammeo and Osmancik untuk mempelajari klasifikasi jenis beras berdasarkan karakteristik morfologi butiran.

## 1. Informasi Proyek

- **Nama Proyek:** Rice Cammeo and Osmancik Classification
- **Mata Kuliah:** Machine Learning
- **Dosen Pengampu:** Asep Erlan Maulana, S.Kom., M.Kom.
- **Nama Mahasiswa:** Muhammad Harist Khairullah
- **NIM:** 241011402909
- **Program Studi:** Teknik Informatika
- **Universitas:** Universitas Pamulang
- **Tahun Akademik:** 2026/2027

## 2. Tujuan Proyek

1. Memahami karakteristik dataset dalam machine learning.
2. Mempelajari proses data loading dan data cleaning menggunakan Python.
3. Memeriksa kualitas data, termasuk nilai kosong dan data duplikat.
4. Melakukan eksplorasi data untuk memahami karakteristik setiap atribut.
5. Mempelajari pembagian dataset untuk pelatihan dan pengujian.
6. Memahami penerapan model klasifikasi untuk membedakan jenis beras.

## 3. Dataset

Dataset yang digunakan adalah **Rice (Cammeo and Osmancik)** dari UCI Machine Learning Repository.

- **Sumber dataset:** https://doi.org/10.24432/C5MW4Z
- **Referensi penelitian:** https://doi.org/10.18201/ijisae.2019355381

Dataset ini berisi karakteristik morfologi butiran beras yang diperoleh melalui pengolahan citra. Data digunakan untuk membedakan dua jenis beras, yaitu Cammeo dan Osmancik.

### Ringkasan Dataset

| Informasi | Keterangan |
|---|---|
| Jumlah baris | 3.810 |
| Jumlah kolom | 8 |
| Jumlah fitur | 7 |
| Kolom target | `Class` |
| Jumlah kelas | 2 |
| Nilai kosong | 0 |
| Data duplikat | 0 |

### Distribusi Kelas

| Kelas | Jumlah |
|---|---:|
| Cammeo | 1.630 |
| Osmancik | 2.180 |
| Total | 3.810 |

## 4. Atribut Dataset

Dataset terdiri dari tujuh fitur dan satu atribut target.

| No. | Atribut | Keterangan |
|---:|---|---|
| 1 | `Area` | Luas butiran beras |
| 2 | `Perimeter` | Keliling butiran beras |
| 3 | `Major_Axis_Length` | Panjang sumbu utama |
| 4 | `Minor_Axis_Length` | Panjang sumbu minor |
| 5 | `Eccentricity` | Tingkat kelonjongan bentuk |
| 6 | `Convex_Area` | Luas area cembung |
| 7 | `Extent` | Rasio area butiran terhadap area pembatasnya |
| 8 | `Class` | Label jenis beras |

Atribut `Class` menjadi target klasifikasi, sedangkan tujuh atribut lainnya digunakan sebagai fitur masukan.

## 5. Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Library tersebut dapat digunakan untuk membaca data, melakukan analisis, membuat visualisasi, membagi dataset, serta membangun dan mengevaluasi model machine learning. Penggunaan aktual setiap library mengikuti program yang dijalankan.

## 6. Tahapan Pengerjaan

### 6.1 Data Loading

Data loading merupakan proses memasukkan dataset ke lingkungan Python agar dapat diperiksa dan diolah. Pada tahap ini, struktur data, jumlah baris, jumlah kolom, dan nama atribut diperiksa untuk memastikan data terbaca dengan benar.

### 6.2 Data Cleaning

Data cleaning dilakukan untuk memeriksa kualitas dataset sebelum analisis. Pemeriksaan meliputi nilai kosong, data duplikat, tipe data, dan konsistensi atribut.

Berdasarkan pemeriksaan dataset yang digunakan, tidak ditemukan nilai kosong maupun baris duplikat.

### 6.3 Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik data melalui statistik deskriptif, distribusi kelas, dan visualisasi atribut. Tahap ini membantu melihat pola serta perbedaan karakteristik pada dataset sebelum pemodelan.

### 6.4 Data Splitting

Data splitting merupakan proses membagi dataset menjadi data pelatihan dan data pengujian. Data pelatihan digunakan untuk membangun model, sedangkan data pengujian digunakan untuk menilai performa model terhadap data yang belum digunakan saat pelatihan.

Rasio pembagian data disesuaikan dengan implementasi program.

### 6.5 Data Modelling

Data modelling merupakan tahap penerapan algoritma machine learning untuk memprediksi jenis beras berdasarkan fitur yang tersedia. Target klasifikasi adalah kelas Cammeo dan Osmancik.

Algoritma dan parameter model harus mengikuti implementasi yang benar-benar digunakan dalam proyek.

### 6.6 Evaluasi Model

Evaluasi model bertujuan untuk mengukur kemampuan model dalam melakukan klasifikasi. Metrik yang dapat digunakan meliputi:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Nilai evaluasi akhir harus diisi berdasarkan hasil pengujian model yang sebenarnya.

## 7. Persiapan Lingkungan

Pastikan Python telah terpasang pada komputer. Selanjutnya, buat virtual environment menggunakan terminal.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instal library yang diperlukan:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

## 8. Membaca Dataset

File yang diberikan memiliki metadata ARFF sebelum bagian data dimulai. Contoh berikut membaca data setelah penanda `@DATA`.

```python
import pandas as pd
from io import StringIO

nama_file = "Rice_Cammeo_Osmancik(1).csv"

with open(nama_file, "r", encoding="utf-8") as file:
    baris = file.readlines()

posisi_data = next(
    i for i, baris_file in enumerate(baris)
    if baris_file.strip().upper() == "@DATA"
)

data = baris[posisi_data + 1:]

df = pd.read_csv(
    StringIO("".join(data)),
    header=None
)

df.columns = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
    "Class"
]

print("Ukuran dataset:", df.shape)
print(df.head())

print("Nilai kosong:", df.isnull().sum().sum())
print("Data duplikat:", df.duplicated().sum())
print(df["Class"].value_counts())
```

**Catatan:** Pastikan nama file dan lokasi dataset sesuai dengan file yang ada di komputer.

## 9. Struktur Folder

Berikut contoh struktur folder yang disarankan untuk repository GitHub.

```text
rice-cammeo-osmancik/
├── README.md
├── Rice_Cammeo_Osmancik(1).csv
├── notebooks/
│   └── analisis_rice.ipynb
└── src/
    └── main.py
```

Struktur tersebut merupakan rekomendasi. Nama dan isi file dapat disesuaikan dengan program yang benar-benar tersedia di repository.

## 10. Hasil yang Diharapkan

Proyek ini diharapkan membantu memahami tahapan pengolahan dataset untuk klasifikasi machine learning, mulai dari data loading, data cleaning, eksplorasi data, pembagian dataset, hingga pemodelan dan evaluasi.

Hasil akhir harus dilengkapi dengan algoritma yang digunakan serta metrik evaluasi dari eksperimen yang benar-benar dijalankan. Nilai akurasi tidak dicantumkan sebelum model diuji.

## 11. Referensi

Cinar, I., & Koklu, M. (2019). Classification of rice varieties using artificial intelligence methods. *International Journal of Intelligent Systems and Applications in Engineering, 7*(3), 188–194. https://doi.org/10.18201/ijisae.2019355381

UCI Machine Learning Repository. (2019). *Rice (Cammeo and Osmancik).* https://doi.org/10.24432/C5MW4Z

## 12. Lisensi dan Atribusi

Dataset berasal dari sumber penelitian yang dicantumkan pada bagian referensi. Periksa ketentuan lisensi dan atribusi pada sumber asli sebelum mendistribusikan kembali dataset.

Dokumentasi ini disusun untuk keperluan praktikum Machine Learning di Program Studi Teknik Informatika, Universitas Pamulang.
