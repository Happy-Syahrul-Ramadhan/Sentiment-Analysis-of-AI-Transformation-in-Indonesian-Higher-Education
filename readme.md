# 🤖 Sentiment Analysis: AI Transformation in Indonesian Higher Education

## 🎨 Web Application

**Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/syahrul19112003/deteksi-sentimen-terhadap-penggunaan-AI-di-lingkungan-perguruan-tinggi)

**Natural Language Processing (NLP)**

---

## 👥 Anggota TIM

| Nama | NIM | Jurusan | GitHub |
|------|-----|---------|--------|
| Happy Syahrul Ramadhan | 122450013 | Sains Data | [@Happy-Syahrul-Ramadhan](https://github.com/Happy-Syahrul-Ramadhan) |
| Ahmad Sahidin Akbar | 122450044 | Sains Data | [@Masahid](https://github.com/AhmadSahidinAkbar) |
| Karin Yehezkiel Sinaga | 123410029 | Sains Aktuaria | [@karinyehzkielsinaga](https://github.com/karinyehzkielsinaga) |

---

## 📝 Deskripsi Proyek

Proyek ini menyelidiki sentimen opini mahasiswa Indonesia terhadap penggunaan Artificial Intelligence (AI) dalam konteks pendidikan tinggi. Menggunakan teknik Natural Language Processing dan Machine Learning, kami membangun model klasifikasi untuk menganalisis apakah opini terhadap penggunaan AI bersifat **Positif**, **Netral**, atau **Negatif**.

### 🎯 Tujuan
- Memahami sentimen dan persepsi mahasiswa terhadap AI di perguruan tinggi
- Mengembangkan model NLP yang akurat untuk klasifikasi sentimen teks bahasa Indonesia
- Mengidentifikasi peluang dan tantangan implementasi AI dalam pendidikan

---

## 📊 Dataset

- **Sumber**: [Sentimen ChatGPT Mahasiswa - Kaggle](https://www.kaggle.com/datasets/bintangaprianta/sentimen-chatgpt-mahasiswa)
- **Jumlah Data**: ~1000+ tweets/opini
- **Label**: 
  - `1` - Positif (mendukung penggunaan AI)
  - `0` - Netral (objektif, tanpa opini jelas)
  - `-1` - Negatif (kritis/khawatir)
- **Bahasa**: Indonesian

---

## 🛠️ Metodologi

### Data Preprocessing
1. **Cleaning**: Hapus URL, mentions, hashtags, punctuation
2. **Normalisasi**: Konversi slang Indonesia ke bahasa baku (e.g., "yg" → "yang")
3. **Tokenization**: Pemisahan teks menjadi token individual
4. **Lowercase**: Standardisasi huruf kecil

```python
# Contoh preprocessing
"Wah, penggunaan ChatGPT di kampus harus diwaspadai!" 
→ "wah penggunaan chatgpt di kampus harus diwaspadai"
```

### Feature Extraction
- **TF-IDF Vectorization**: Mengkonversi teks ke vektor numerik
- **N-grams**: Unigram & Bigram untuk capturing context

### Model Classification
- **Algorithm**: Support Vector Machine (SVM)
- **Library**: scikit-learn
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

---

## � Model Performance & Training Results

### 🏆 Final Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **82.14%** |
| **Precision** | **82.14%** |
| **Recall** | **82.14%** |
| **F1-Score** | **82.14%** |
| **Best Model** | **SVM** (Val Acc: 87.25%) |

### 🚀 SENTIMENT ANALYSIS - TRAINING WITH COMBINED DATA

#### 📄 Data Loading & Preparation

```
Loading: combined_training_data.csv
   ✓ Encoding: utf-8
   Total rows: 2,295
```

**Class Distribution:**
| Kelas | Count | Persentase |
|-------|-------|-----------|
| Negative | 1,158 | 50.5% |
| Neutral | 0 | 0.0% |
| Positive | 1,137 | 49.5% |

**Data Split (Train:Val:Test = 80:10:10):**
- Train: 1,632 samples (80%)
- Val: 204 samples (10%)
- Test: 459 samples (10%)

#### 📝 Feature Extraction

```
TF-IDF Vectorization:
   ✓ Feature matrix shape: (1,632, 2,257)
   ✓ N-grams: Unigram & Bigram
```

#### ▶️ Model Training Comparison

| Model | Validation Accuracy | Training Time |
|-------|-------------------|---------------|
| Naive Bayes | 84.80% | 0.00s |
| Random Forest | 62.25% | 0.19s |
| **SVM ⭐** | **87.25%** | **0.15s** |

#### 🧪 Test Set Evaluation

**Detailed Classification Report:**
```
                precision    recall  f1-score   support

    Negative       0.83      0.82      0.82       232
    Positive       0.82      0.82      0.82       227

    accuracy                           0.82       459
   macro avg       0.82      0.82      0.82       459
weighted avg       0.82      0.82      0.82       459
```

---

## �📁 Struktur Proyek

```
sentimen analisis/
├── Deep Learning/                  # Deep Learning module with full training
│   ├── data/
│   │   ├── combined_training_data.csv    # Combined dataset (2,295 rows)
│   │   ├── data_ai.csv
│   │   ├── negasi.txt
│   │   ├── s-neg.txt
│   │   └── s-pos.txt
│   ├── models/
│   │   ├── sentiment_classifier.pkl     # Trained classifier model
│   │   └── vectorizer.pkl               # TF-IDF vectorizer
│   ├── plots/
│   │   └── training_metrics.csv
│   ├── clean_combined_data.py
│   ├── clean_data.py
│   ├── data_preprocessing.py
│   ├── datareader.py
│   ├── model.py
│   ├── remove_neutral.py
│   ├── train.py
│   ├── training.py
│   └── utils.py
├── data/
│   └── data_ai.xlsx              # Original dataset
├── notebook/
│   └── sentiment_analisis_about_AI_in_Academic.ipynb
├── deteksi-opini-terhadap-penggunaan-ai/
│   ├── app.py                    # Gradio web app
│   ├── requirements.txt
│   └── README.md
└── readme.md                      # Dokumentasi proyek
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Happy-Syahrul-Ramadhan/Sentiment-Analysis-of-AI-Transformation-in-Indonesian-Higher-Education.git
cd "sentimen analisis"
```

### 2. Setup Environment
```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
cd deteksi-opini-terhadap-penggunaan-ai
pip install -r requirements.txt
```

### 3. Run App Locally
```bash
python app.py
# Akses di: http://localhost:7860
```

### 4. Run Deep Learning Training (Optional)
```bash
cd "Deep Learning"
python training.py
# Output: trained models, metrics, dan evaluation results
```

---

### Fitur
- ✅ Input teks bebas tentang opini AI
- ✅ Prediksi sentimen real-time
- ✅ Confidence score untuk setiap kategori
- ✅ Contoh teks pre-built untuk testing

### Stack Teknologi
- **Frontend**: Gradio (UI interaktif)
- **Backend**: Python (scikit-learn)
- **Deployment**: Hugging Face Spaces


## 📚 Dependencies

```
gradio>=4.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
openpyxl  # untuk read Excel files
```



## 📖 Referensi & Sumber Daya

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Gradio Documentation](https://www.gradio.app/)
- [Indonesian NLP Resources](https://github.com/NoveliBrains/Indonesian-NLP)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## 📄 Lisensi

Project ini menggunakan lisensi **MIT**. Lihat file [LICENSE](LICENSE) untuk informasi lengkap.

---

## 📞 Kontak & Support

Jika ada pertanyaan atau masukan, silakan buat issue di repository ini atau hubungi anggota kelompok.

