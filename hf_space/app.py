"""
app.py — Gradio App untuk Analisis Sentimen Opini AI di Perguruan Tinggi
=========================================================================
Deploy di Hugging Face Spaces.
Model: SVM Classification Pipeline (.pkl)
"""

print("Starting app initialization...")
import re
import pickle
print("Importing Gradio...")
import gradio as gr
print("Importing Pandas...")
import pandas as pd

# ══════════════════════════════════════════════
# 📦 LOAD MODEL & VECTORIZER
# ══════════════════════════════════════════════
print("Loading Model and Vectorizer...")
model = None
vectorizer = None

try:
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✓ SVM Model Successfully Loaded!")
except Exception as e:
    print(f"✗ Error loading SVM model: {e}")

try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✓ TF-IDF Vectorizer Successfully Loaded!")
except Exception as e:
    print(f"✗ Error loading vectorizer: {e}")

# ══════════════════════════════════════════════
# 🔤 PREPROCESSING (sama persis dengan training)
# ══════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """
    Pipeline pembersihan teks untuk analisis sentimen AI di perguruan tinggi
    (sama persis dengan preprocessing di notebook)
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus URL/link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Hapus mention (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Normalize hashtag (hapus tanda # tapi tetap simpan kata)
    text = re.sub(r'#', '', text)
    
    # Hapus tanda baca dan angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)      # Hapus angka
    
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_slang(text: str) -> str:
    """Normalisasi slang Indonesia umum"""
    slang_map = {
        'yg': 'yang', 'sgt': 'sangat', 'kl': 'kalau', 'klo': 'kalau',
        'dlm': 'dalam', 'krn': 'karena', 'tp': 'tapi', 'gpp': 'gapapa',
        'u': 'anda', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
        'lo': 'anda', 'lu': 'anda', 'elu': 'anda',
        'dg': 'dengan', 'kek': 'seperti', 'ato': 'atau',
        'udh': 'sudah', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
        'bgd': 'banget', 'bgt': 'banget', 'dr': 'dari',
        'skrg': 'sekarang', 'kmrn': 'kemarin', 'tdi': 'tadi',
        'lg': 'lagi', 'sdg': 'sedang', 'tgs': 'tugas',
        'uts': 'ujian tengah semester', 'uas': 'ujian akhir semester',
        'chatgptnya': 'chatgpt', 'gpt': 'chatgpt',
    }
    
    words = text.split()
    normalized_words = [slang_map.get(word, word) for word in words]
    
    # Handle repeated characters (jangan terlalu aggressive)
    normalized = []
    for word in normalized_words:
        word = re.sub(r'([a-z])\1{2,}', r'\1\1', word)
        normalized.append(word)
    
    return ' '.join(normalized)


def clean_text(text: str) -> str:
    """Pipeline pembersihan lengkap"""
    text = preprocess_text(text)
    text = normalize_slang(text)
    return text


# ══════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════

def predict_sentiment(text: str) -> dict:
    """
    Prediksi sentimen opini terhadap AI dari teks.
    
    Returns:
        Dictionary {label: confidence} untuk Gradio Label component.
    """
    if model is None or vectorizer is None:
        missing = []
        if model is None:
            missing.append("SVM model")
        if vectorizer is None:
            missing.append("TF-IDF vectorizer")
        return {f"Error: {', '.join(missing)} tidak berhasil dimuat": 1.0}
    
    if not text or not text.strip():
        return {"Error: Input teks kosong": 1.0}
    
    try:
        # 1. Bersihkan teks
        cleaned = clean_text(text)
        
        if not cleaned:
            return {"Teks kosong setelah dibersihkan": 1.0}
        
        # 2. Vectorize teks menggunakan TF-IDF
        vectorized = vectorizer.transform([cleaned])
        
        # 3. Prediksi menggunakan SVM model
        prediction = model.predict(vectorized)[0]
        confidence = max(model.predict_proba(vectorized)[0]) if hasattr(model, 'predict_proba') else 1.0
        
        # 4. Map label numerik ke string
        label_map = {-1: 'Negatif', 0: 'Netral', 1: 'Positif'}
        label = label_map.get(int(prediction), str(prediction))
        
        return {label: float(confidence)}
    
    except Exception as e:
        return {f"Error: {str(e)}": 1.0}


# ══════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════

EXAMPLES = [
    ["AI sangat membantu dalam proses pembelajaran dan meningkatkan efisiensi akademik mahasiswa"],
    ["Penggunaan ChatGPT membuat mahasiswa malas berpikir kritis sendiri dan hanya copy paste"],
    ["Tools AI seperti GitHub Copilot dan ChatGPT cukup berguna untuk membantu pemrograman"],
    ["AI akan menggantikan peran dosen universitas, ini adalah kecemasan yang serius"],
    ["Teknologi machine learning membuka peluang penelitian yang sangat menarik di kampus"],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="💬 Masukkan Opini Anda",
        placeholder="Ketik opini tentang penggunaan AI di perguruan tinggi...",
        lines=4,
    ),
    outputs=gr.Label(
        label="🎯 Hasil Prediksi Sentimen",
        num_top_classes=3,
    ),
    title="🤖 Analisis Sentimen Opini AI di Perguruan Tinggi",
    description=(
        "Model NLP untuk mendeteksi sentimen opini terhadap penggunaan AI "
        "dalam pendidikan tinggi Indonesia.\n\n"
        "Klasifikasi: **Positif** (mendukung) | **Netral** (objektif) | **Negatif** (kritis)\n\n"
        "Model: SVM\n"
        "Preprocessing: TF-IDF Vectorization + Custom text cleaning untuk konteks akademik Indonesia."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()