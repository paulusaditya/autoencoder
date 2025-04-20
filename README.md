# 🧠 Autoencoder untuk Kompresi dan Rekonstruksi Ekspresi Wajah

Proyek ini menggunakan autoencoder berbasis CNN (Convolutional Neural Network) untuk merekonstruksi gambar wajah dari dataset ekspresi wajah manusia. Model dilatih untuk mengenali pola utama dari citra dan menghasilkan output yang mirip dengan input (self-supervised learning).

---

## 📁 Dataset

Dataset berisi gambar ekspresi wajah manusia dengan 7 kategori emosi:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

Struktur folder dataset:

dataset/ ├── train/ │ ├── angry/ │ ├── disgust/ │ ├── fear/ │ ├── happy/ │ ├── neutral/ │ ├── sad/ │ └── surprise/ └── test/ ├── angry/ ├── disgust/ ├── fear/ ├── happy/ ├── neutral/ ├── sad/ └── surprise/


- Jumlah data: ~57.000 gambar
- **Format**: JPG
- **Resolusi**: 48x48 piksel (akan diskalakan otomatis jika berbeda)

Contoh nama file:
PrivateTest_88305.jpg PrivateTest_7543291.jpg


> 📌 Note: Jika format gambar bukan grayscale, pipeline preprocessing akan otomatis mengkonversi ke grayscale saat loading data.

---

## 🧬 Arsitektur Autoencoder

Model autoencoder dibangun menggunakan Keras (TensorFlow backend). Berikut ringkasan arsitekturnya:

### Encoder
- Conv2D (32 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3) + ReLU
- MaxPooling2D (2x2)

### Decoder
- Conv2D (64 filters, 3x3) + ReLU
- UpSampling2D (2x2)
- Conv2D (32 filters, 3x3) + ReLU
- UpSampling2D (2x2)
- Conv2D (1 filter, 3x3, activation='sigmoid')

---

## 📊 Hasil Training

Model dilatih selama 50 epoch menggunakan loss function `binary_crossentropy` dan optimizer `Adam`.

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 0.5611     | 0.5531   |
| 10    | 0.5479     | 0.5473   |
| 25    | 0.5469     | 0.5462   |
| 50    | 0.5465     | 0.5458   |

Loss model menunjukkan konvergensi yang baik antara training dan validation.

---

## 🖼️ Contoh Input dan Output

Berikut perbandingan antara gambar input dan hasil rekonstruksi dari autoencoder:

| Input                          | Rekonstruksi                      |
|--------------------------------|-----------------------------------|
| ![Input](samples/input1.jpg)   | ![Output](samples/output1.jpg)   |
| ![Input](samples/input2.jpg)   | ![Output](samples/output2.jpg)   |
| ![Input](samples/input3.jpg)   | ![Output](samples/output3.jpg)   |

> 📌 Hasil rekonstruksi mempertahankan bentuk umum wajah dan ekspresi, meskipun terdapat sedikit kehilangan detail halus.

---

## 🚀 Cara Menjalankan

1. **Clone repo**:
```bash
git clone https://github.com/paulusaditya/autoencoder
cd autoencoder

## Install dependencies:

bash
Salin
Edit
pip install -r requirements.txt
Jalankan training:

bash
Salin
Edit
python run.py
Lihat hasil rekonstruksi di folder results/.

## 🛠 Teknologi yang Digunakan
Python 3.x
TensorFlow / Keras
NumPy, Matplotlib
OpenCV (untuk pre-processing gambar)

## 📚 Referensi
Deep Learning with Python - François Chollet
FER2013 Dataset
Paper: Autoencoders, Unsupervised Learning, and Deep Architectures - Bengio et al.
