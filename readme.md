# Sentiment Analysis of AI Adoption in Indonesian Higher Education

Repositori ini berisi implementasi penelitian **"Sentiment Analysis of AI Adoption in Indonesian Higher Education Using Machine Learning and Transformer-Based Models"**. Proyek ini membandingkan pendekatan machine learning berbasis **TF-IDF** dengan model deep learning berbasis **DistilBERT** untuk klasifikasi sentimen opini mahasiswa Indonesia terhadap penggunaan Artificial Intelligence (AI) di lingkungan perguruan tinggi.

## Link Paper
https://arxiv.org/pdf/2604.27439

## Web Application

**Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/syahrul19112003/deteksi-sentimen-terhadap-penggunaan-AI-di-lingkungan-perguruan-tinggi)

Aplikasi web memungkinkan pengguna memasukkan teks opini tentang penggunaan AI di perguruan tinggi dan mendapatkan prediksi sentimen secara real-time.

## Anggota Tim

| Nama | Afiliasi | Email / GitHub |
|------|----------|----------------|
| Happy Syahrul Ramadhan | Faculty of Science, Sumatra Institute of Technology | happy.122450013@student.itera.ac.id / [@Happy-Syahrul-Ramadhan](https://github.com/Happy-Syahrul-Ramadhan) |
| Ahmad Sahidin Akbar | Faculty of Science, Sumatra Institute of Technology | ahmad.122450044@student.itera.ac.id / [@Masahid](https://github.com/AhmadSahidinAkbar) |
| Karin Yehezkiel Sinaga | Faculty of Science, Sumatra Institute of Technology | karin.123410029@student.itera.ac.id / [@karinyehzkielsinaga](https://github.com/karinyehzkielsinaga) |

## Project Overview

Penggunaan AI dalam pendidikan tinggi menghadirkan peluang seperti peningkatan efisiensi belajar, akses informasi yang lebih luas, dan dukungan terhadap fleksibilitas akademik. Namun, adopsi AI juga menimbulkan kekhawatiran terkait penyalahgunaan, ketergantungan terhadap keluaran AI, serta potensi penurunan kemampuan berpikir kritis mahasiswa.

Proyek ini menganalisis opini mahasiswa Indonesia terhadap penggunaan AI di perguruan tinggi melalui klasifikasi sentimen biner:

- **Positive**: opini yang mendukung atau menilai positif penggunaan AI.
- **Negative**: opini yang kritis, khawatir, atau menilai negatif penggunaan AI.

## Research Objectives

- Menganalisis sentimen mahasiswa Indonesia terhadap adopsi AI di perguruan tinggi.
- Membandingkan performa model machine learning klasik dan Transformer-based deep learning.
- Mengevaluasi efektivitas **SVM dengan TF-IDF** sebagai baseline yang efisien.
- Mengevaluasi kemampuan **DistilBERT** dalam menangkap konteks bahasa untuk klasifikasi sentimen.
- Menyediakan aplikasi web interaktif untuk prediksi sentimen secara real-time.

## Dataset

Dataset akhir terdiri dari **2,295 sampel** untuk klasifikasi sentimen biner. Dataset dibentuk dari gabungan opini mahasiswa tentang AI dan data leksikal sentimen positif-negatif.

| Data Source | Number of Samples | Description |
|-------------|------------------:|-------------|
| Student Opinions on AI | 1,154 | Main research data |
| Positive and Negative Lexical Dictionary | 1,141 | Additional sentiment-based data |
| **Total** | **2,295** | Final dataset |

Distribusi kelas pada dataset akhir relatif seimbang.

| Class | Number of Samples | Percentage |
|-------|------------------:|-----------:|
| Negative | 1,158 | 50.46% |
| Positive | 1,137 | 49.54% |
| **Total** | **2,295** | **100%** |

Data dibagi menjadi training, validation, dan test set dengan rasio 80:10:10.

| Subset | Number of Samples | Percentage |
|--------|------------------:|-----------:|
| Training Data | 1,632 | 80% |
| Validation Data | 204 | 10% |
| Test Data | 459 | 10% |

## Research Workflow

Alur penelitian terdiri dari beberapa tahap utama:

1. Data loading.
2. Data cleaning dan preprocessing.
3. Data splitting.
4. Machine learning branch: TF-IDF, model training, model selection, dan evaluation.
5. Deep learning branch: DistilBERT tokenizer, fine-tuned DistilBERT, dan evaluation.
6. Model comparison.
7. Deployment ke aplikasi web.

## Preprocessing

Tahap preprocessing dilakukan untuk membersihkan dan menstandarkan teks sebelum digunakan dalam pelatihan model.

Langkah-langkah preprocessing meliputi:

- Lowercasing.
- Menghapus URL.
- Menghapus mentions.
- Menghapus hashtags.
- Menghapus punctuation.
- Menghapus extra whitespace.
- Normalisasi kata tidak baku.
- Normalisasi karakter berulang.

Contoh:

```text
Input  : Wah, penggunaan ChatGPT di kampus harus diwaspadai!
Output : wah penggunaan chatgpt di kampus harus diwaspadai
```

## Methodology

Penelitian ini menggunakan dua pendekatan utama untuk klasifikasi sentimen.

### 1. Machine Learning with TF-IDF

Pada pendekatan machine learning, teks yang telah diproses dikonversi menjadi fitur numerik menggunakan **TF-IDF** dengan unigram dan bigram. TF-IDF vectorizer hanya di-fit pada data training untuk menghindari data leakage.

Model yang dibandingkan:

| Model | Configuration | Main Parameter | Reason for Use |
|-------|---------------|----------------|----------------|
| LightGBM | `LGBMClassifier` | Boosting ensemble | Fast and efficient initial baseline |
| Random Forest | `RandomForestClassifier` | Parallel ensemble | Robust and able to reduce overfitting through averaging |
| SVM | `SVC(kernel='rbf', C=1.0)` | Kernel-based classifier | Suitable for high-dimensional text data and non-linear separation |

Detail parameter:

- **LightGBM**: `n_estimators=100`, `max_depth=10`, `learning_rate=0.1`
- **Random Forest**: `n_estimators=100`, `max_depth=20`, `min_samples_split=5`
- **SVM**: RBF kernel, `C=1.0`
- Semua model menggunakan `random_state=42` untuk reproducibility.

### 2. Transformer-Based Deep Learning with DistilBERT

Pendekatan deep learning menggunakan **DistilBERT** untuk menangkap informasi kontekstual dari teks. Input teks ditokenisasi menjadi `input_ids` dan `attention_mask`, kemudian diproses oleh encoder DistilBERT dan classification head untuk prediksi sentimen negatif atau positif.

Konfigurasi DistilBERT:

| Parameter | Value |
|----------|-------|
| Model | `distilbert-base-uncased` |
| Max Length | 128 |
| Vocabulary Size | 30,522 |
| Batch Size | 64 |
| Learning Rate | 1e-6 |
| Epochs | 100 |
| Early Stopping Patience | 5 |
| Weight Decay | 0.001 |
| Dropout | 0.5 |
| Warmup Steps | 100 |
| Optimizer | AdamW |
| Scheduler | Linear |
| Best Metric | Validation F1-score |
| Device | CUDA |

## Evaluation Metrics

Model dievaluasi menggunakan metrik berikut:

- **Accuracy**: proporsi prediksi benar dari seluruh data.
- **Precision**: ketepatan model dalam memprediksi kelas positif.
- **Recall**: kemampuan model mengenali sampel positif aktual.
- **F1-score**: harmonic mean antara precision dan recall.

## Results

### Machine Learning Results

| Model | Approach | Validation Accuracy | Test Accuracy | Test F1-score | Training Time |
|-------|----------|--------------------:|--------------:|--------------:|--------------:|
| LightGBM | Machine Learning | 57.35% | 49.67% | 48.46% | 0.057s |
| Random Forest | Machine Learning | 62.25% | 58.61% | 56.48% | 0.216s |
| SVM | Machine Learning | **87.25%** | **82.14%** | **82.14%** | 0.210s |

SVM menjadi model machine learning terbaik. Model ini berhasil mengklasifikasikan **377 dari 459** sampel test dengan benar, terdiri dari 190 sampel negatif dan 187 sampel positif. Distribusi kesalahan relatif seimbang, sehingga performanya stabil pada kedua kelas.

### DistilBERT Results

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| Negative | 0.8240 | 0.8879 | 0.8548 | 116 |
| Positive | 0.8762 | 0.8070 | 0.8402 | 114 |
| **Accuracy** |  |  | **0.8478** | 230 |
| **Macro Avg** | 0.8501 | 0.8475 | 0.8475 | 230 |
| **Weighted Avg** | 0.8499 | 0.8478 | 0.8475 | 230 |

DistilBERT menghasilkan performa test terbaik secara keseluruhan dengan:

- **Test Accuracy**: 84.78%
- **Weighted F1-score**: 84.75%
- **Final Train Loss**: 0.2827
- **Final Validation Loss**: 0.3649

Model lebih sensitif terhadap kelas negatif dan lebih presisi pada prediksi kelas positif, tetapi F1-score kedua kelas tetap relatif seimbang.

### Final Model Comparison

| Model | Approach | Validation Accuracy | Test Accuracy | Test F1-score | Training Time |
|-------|----------|--------------------:|--------------:|--------------:|--------------:|
| LightGBM | Machine Learning | 57.35% | 49.67% | 48.46% | 0.057s |
| Random Forest | Machine Learning | 62.25% | 58.61% | 56.48% | 0.216s |
| SVM | Machine Learning | **87.25%** | 82.14% | 82.14% | **0.210s** |
| DistilBERT | Transformer-based Deep Learning | 81.74% | **84.78%** | **84.75%** | 22.8m |

### Key Findings

- **DistilBERT** memberikan performa test terbaik dengan accuracy 84.78% dan weighted F1-score 84.75%.
- **SVM dengan TF-IDF** menjadi model machine learning terbaik dengan test accuracy dan F1-score 82.14%.
- DistilBERT mengungguli SVM sebesar **2.64% pada accuracy** dan **2.61% pada F1-score**.
- SVM tetap menarik untuk penggunaan dengan keterbatasan resource karena training jauh lebih cepat.
- LightGBM dan Random Forest kurang optimal untuk sparse high-dimensional TF-IDF features dibandingkan SVM.

## Deployment

Model terbaik dideploy sebagai aplikasi web interaktif menggunakan **Gradio** dan **Hugging Face Spaces**. Aplikasi memungkinkan pengguna memasukkan teks opini dan menerima prediksi sentimen beserta confidence score.

### Features

- Input teks bebas tentang opini penggunaan AI di perguruan tinggi.
- Prediksi sentimen biner: positive atau negative.
- Confidence score untuk hasil prediksi.
- Interface sederhana berbasis Gradio.
- Deployment online melalui Hugging Face Spaces.

### Technology Stack

- **Language**: Python
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch, Transformers
- **Data Processing**: pandas, numpy
- **Web Interface**: Gradio
- **Deployment**: Hugging Face Spaces

## Repository Structure

```text
sentimen analisis/
├── Deep Learning/
│   ├── data/
│   │   ├── combined_training_data.csv
│   │   ├── data_ai.csv
│   │   ├── negasi.txt
│   │   ├── s-neg.txt
│   │   └── s-pos.txt
│   ├── models/
│   │   ├── sentiment_classifier.pkl
│   │   └── vectorizer.pkl
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
│   └── data_ai.xlsx
├── notebook/
│   └── sentiment_analisis_about_AI_in_Academic.ipynb
├── deteksi-opini-terhadap-penggunaan-ai/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Happy-Syahrul-Ramadhan/Sentiment-Analysis-of-AI-Transformation-in-Indonesian-Higher-Education.git
cd "sentimen analisis"
```

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r deteksi-opini-terhadap-penggunaan-ai/requirements.txt
```

### 3. Run Web App Locally

```bash
cd deteksi-opini-terhadap-penggunaan-ai
python app.py
```

Akses aplikasi di:

```text
http://localhost:7860
```

### 4. Run Machine Learning Training

```bash
cd "Deep Learning"
python training.py
```

### 5. Run DistilBERT Training

```bash
cd "Deep Learning"
python train.py
```

> Catatan: Training DistilBERT disarankan menggunakan CUDA/GPU untuk waktu komputasi yang lebih efisien.

## Dependencies

```text
# Web App
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl

# Machine Learning
scikit-learn>=1.3.0
lightgbm

# Deep Learning
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
```

## Citation

Jika menggunakan kode atau hasil dari proyek ini, silakan sitasi paper berikut:

```bibtex
@article{ramadhan2026sentiment,
  title   = {Sentiment Analysis of AI Adoption in Indonesian Higher Education Using Machine Learning and Transformer-Based Models},
  author  = {Ramadhan, Happy Syahrul and Akbar, Ahmad Sahidin and Sinaga, Karin Yehezkiel and Muthoharoh, Luluk and Satria, Ardika and Manullang, Martin C. T.},
  year    = {2026},
  journal = {arXiv preprint arXiv:2604.27439}
}
```

## References

- Bing Liu. *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers, 2012.
- Ashish Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017.
- Walaa Medhat, Ahmed Hassan, and Hoda Korashy. *Sentiment analysis algorithms and applications: A survey*. Ain Shams Engineering Journal, 2014.
- Bryan Wilie et al. *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. AACL-IJCNLP, 2020.
- Victor Sanh et al. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS EMC2 Workshop, 2019.
- Jacob Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT, 2019.
- Samuel Cahyawijaya et al. *NusaCrowd: Open Source Initiative for Indonesian NLP Resources*. ACL Findings, 2023.

## License

Project ini menggunakan lisensi **MIT**. Lihat file [LICENSE](LICENSE) untuk informasi lengkap.

## Contact

Jika ada pertanyaan atau masukan, silakan buat issue di repository ini atau hubungi anggota tim.
