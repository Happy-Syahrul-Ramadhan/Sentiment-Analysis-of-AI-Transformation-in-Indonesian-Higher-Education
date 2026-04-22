"""
datareader.py — Download, Preprocessing, Dataset, dan DataLoader
=================================================================
Alur data untuk DistilBERT:
  1. download_dataset()  — unduh CSV dari Kaggle via kagglehub
  2. load_and_clean()    — baca CSV, bersihkan teks, encode label
  3. BERTDataset         — PyTorch Dataset: tokenisasi per sampel
  4. get_dataloaders()   — bagi data train/val/test, buat DataLoader
"""

import os
import re
import shutil

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

CSV_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")

TEXT_COL  = "isi_tweet"
LABEL_COL = "label"

# Label: -1 (negatif), 1 (positif) — Binary classification
LABEL_MAP = {-1: "negative", 1: "positive"}
LABEL_LIST = ["negative", "positive"]
NUM_CLASSES = 2

BERT_MODEL   = "distilbert-base-uncased"
BERT_MAX_LEN = 128
BATCH_SIZE   = 64
TEST_SIZE    = 0.10
VAL_SIZE     = 0.10
SAMPLE_SIZE  = None    # None = pakai semua data
RANDOM_SEED  = 42

# Buat folder jika belum ada
for _dir in (DATA_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(_dir, exist_ok=True)


# ──────────────────────────────────────────────
# 1. DOWNLOAD DATASET
# ──────────────────────────────────────────────

def download_dataset() -> str:
    """
    Pastikan dataset tersedia di DATA_DIR.
    Dataset sudah ada di folder data/, jadi fungsi ini hanya memverifikasi.

    Returns:
        Path ke file CSV.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {CSV_PATH}\n"
            "Pastikan file data_ai.csv ada di folder data/."
        )
    
    print(f"Dataset ditemukan: {CSV_PATH}")
    return CSV_PATH


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase → hapus URL & HTML → hapus non-alpha → strip spasi."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(csv_path: str, use_lexicon: bool = False) -> tuple[list, list, LabelEncoder]:
    """
    Baca combined CSV (yang sudah include lexicon data), bersihkan teks, encode label.

    Args:
        csv_path: Path ke file CSV (combined_training_data.csv)
        use_lexicon: Deprecated - data sudah combined di CSV

    Returns:
        texts  : list string teks bersih
        labels : list integer label
        le     : LabelEncoder (digunakan untuk decode prediksi)
    """
    print("=" * 70)
    print("LOADING COMBINED TRAINING DATA")
    print("=" * 70)
    
    # ─── Load combined CSV data ───
    print(f"\n📄 Reading CSV: {csv_path}")
    
    # Try different encodings
    df = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"   ✓ Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Failed to read {csv_path} with any encoding")
    
    df.columns = df.columns.str.strip()
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"}, inplace=True)
    df.dropna(inplace=True)

    # Konversi label numerik ke string untuk consistency
    df["label"] = df["label"].astype(int).map(LABEL_MAP)
    df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)
    print(f"   ✓ Total samples: {len(df):,}")

    # Sampling stratified (opsional, untuk mempercepat demo)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df, _ = train_test_split(
            df, train_size=SAMPLE_SIZE,
            stratify=df["label"], random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)
        print(f"   ✓ After sampling: {len(df):,}")

    # Bersihkan teks
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Encode label ke integer
    le = LabelEncoder()
    le.fit(LABEL_LIST)
    labels = le.transform(df["label"]).tolist()
    texts  = df["text"].tolist()

    # Summary
    print(f"\n📊 Class distribution:")
    for label_str, count in df["label"].value_counts().items():
        print(f"   {label_str:10s}: {count:6,d} ({100*count/len(df):5.1f}%)")
    print()
    
    return texts, labels, le


# ──────────────────────────────────────────────
# 3. PYTORCH DATASET
# ──────────────────────────────────────────────

class BERTDataset(Dataset):
    """
    Dataset PyTorch untuk DistilBERT.
    Tokenisasi dilakukan saat __getitem__ dipanggil (lazy).
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = BERT_MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 4. DATALOADER BUILDER
# ──────────────────────────────────────────────

def get_dataloaders(
    texts: list,
    labels: list,
    tokenizer,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data menjadi train/val/test lalu buat DataLoaders.

    Split stratified: 80% train / 10% val / 10% test.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Pisahkan test set dulu
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Pisahkan val dari sisa train
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    print(f"Split data — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    train_ds = BERTDataset(X_train, y_train, tokenizer)
    val_ds   = BERTDataset(X_val,   y_val,   tokenizer)
    test_ds  = BERTDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
