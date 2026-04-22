"""
data_preprocessing.py — Complete Data Preprocessing untuk Training
===================================================================
Pipeline preprocessing:
  1. Load data
  2. Convert to lowercase
  3. Remove URLs, mentions, hashtags
  4. Remove punctuation & special characters
  5. Remove extra whitespace
  6. Remove empty texts
  7. Handle duplicates
  8. Save cleaned data
"""

import os
import re
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")


def preprocess_text(text):
    """
    Comprehensive text preprocessing.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions (@username)
    4. Remove hashtags (#tag)
    5. Remove HTML tags
    6. Remove numbers
    7. Remove punctuation & special chars (keep only a-z and spaces)
    8. Remove extra whitespace
    9. Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove hashtags (#tag)
    text = re.sub(r'#\w+', '', text)
    
    # 5. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Remove punctuation & special chars (keep only a-z and spaces)
    # Keep: a-z, space, hyphen
    text = re.sub(r'[^a-z\s\-]', '', text)
    
    # 8. Remove repeated characters (e.g., "aaaaaa" → "a")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 9. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 10. Strip leading/trailing spaces
    text = text.strip()
    
    return text


def preprocess_data():
    """Complete preprocessing pipeline."""
    print("=" * 90)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 90)
    
    # ─── LOAD DATA ───
    print("\n📂 STEP 1: Loading data...")
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(INPUT_PATH, encoding=encoding)
            print(f"   ✓ File loaded (encoding: {encoding})")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Failed to read CSV")
    
    print(f"   Initial rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # ─── DATA QUALITY CHECK ───
    print(f"\n🔍 STEP 2: Data quality check...")
    print(f"   Missing values:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"      - {col}: {missing} ({100*missing/len(df):.1f}%)")
    
    # Drop rows with missing values
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"   ✓ Dropped {dropped:,} rows with missing values")
    
    # ─── TEXT PREPROCESSING ───
    print(f"\n✏️  STEP 3: Text preprocessing...")
    print(f"   Operations:")
    print(f"      • Convert to lowercase")
    print(f"      • Remove URLs")
    print(f"      • Remove mentions (@username)")
    print(f"      • Remove hashtags (#tag)")
    print(f"      • Remove HTML tags")
    print(f"      • Remove numbers")
    print(f"      • Remove punctuation & special characters")
    print(f"      • Remove repeated characters")
    print(f"      • Clean whitespace")
    
    df['isi_tweet'] = df['isi_tweet'].apply(preprocess_text)
    print(f"   ✓ Preprocessing completed")
    
    # ─── REMOVE EMPTY TEXTS ───
    print(f"\n🗑️  STEP 4: Removing empty texts...")
    initial_len = len(df)
    df = df[df['isi_tweet'].str.len() > 0].reset_index(drop=True)
    removed = initial_len - len(df)
    print(f"   Removed: {removed:,} empty texts")
    print(f"   Remaining: {len(df):,} texts")
    
    # ─── REMOVE SHORT TEXTS ───
    print(f"\n📏 STEP 5: Removing too short texts (< 5 chars)...")
    initial_len = len(df)
    df = df[df['isi_tweet'].str.len() >= 5].reset_index(drop=True)
    removed = initial_len - len(df)
    print(f"   Removed: {removed:,} short texts")
    print(f"   Remaining: {len(df):,} texts")
    
    # ─── REMOVE DUPLICATES ───
    print(f"\n🔗 STEP 6: Removing duplicates...")
    initial_len = len(df)
    df = df.drop_duplicates(subset=['isi_tweet']).reset_index(drop=True)
    removed = initial_len - len(df)
    print(f"   Removed: {removed:,} duplicate texts")
    print(f"   Remaining: {len(df):,} texts")
    
    # ─── BALANCE DATASET ───
    print(f"\n⚖️  STEP 7: Checking class balance...")
    for label, name in [(-1, "Negative"), (0, "Neutral"), (1, "Positive")]:
        count = len(df[df['label'] == label])
        pct = 100 * count / len(df)
        print(f"   {name:10s}: {count:6,d} ({pct:5.1f}%)")
    
    # ─── SAMPLE DISPLAY ───
    print(f"\n📋 STEP 8: Sample preprocessed texts:")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        text_display = row['isi_tweet'][:80] + ("..." if len(row['isi_tweet']) > 80 else "")
        label_name = {-1: "Negative", 0: "Neutral", 1: "Positive"}.get(int(row['label']), "Unknown")
        print(f"\n   [{idx+1}] {label_name}")
        print(f"       \"{text_display}\"")
    
    # ─── SAVE ───
    print(f"\n💾 STEP 9: Saving cleaned data...")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"   ✓ Saved to: {OUTPUT_PATH}")
    print(f"   Total samples: {len(df):,}")
    
    # ─── FINAL SUMMARY ───
    print(f"\n" + "=" * 90)
    print("✅ DATA PREPROCESSING COMPLETED")
    print("=" * 90)
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {', '.join(df.columns.tolist())}")
    print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print(f"\n✓ Data is ready for training!")
    
    return df


if __name__ == "__main__":
    df_cleaned = preprocess_data()
