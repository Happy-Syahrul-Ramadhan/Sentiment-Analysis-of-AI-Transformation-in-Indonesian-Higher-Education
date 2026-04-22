"""
clean_combined_data.py — Bersihkan combined data (hapus kolom tidak perlu)
===========================================================================
Hapus kolom: penulis, id_user, waktu
Simpan hanya: isi_tweet, label
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")


def clean_data():
    """Bersihkan data combined."""
    print("=" * 80)
    print("CLEANING COMBINED DATA")
    print("=" * 80)
    
    # Load
    print(f"\n📄 Loading: {INPUT_PATH}")
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(INPUT_PATH, encoding=encoding)
            print(f"   ✓ Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Failed to read CSV")
    
    print(f"   Original columns: {list(df.columns)}")
    print(f"   Total rows: {len(df):,}")
    
    # Check which columns exist
    columns_to_keep = ['isi_tweet', 'label']
    columns_to_drop = []
    
    for col in ['penulis', 'id_user', 'waktu', 'tanggal', 'author', 'user_id', 'time', 'date']:
        if col in df.columns:
            columns_to_drop.append(col)
    
    print(f"\n🗑️  Columns to remove: {columns_to_drop}")
    print(f"   Columns to keep: {columns_to_keep}")
    
    # Drop unnecessary columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # Keep only isi_tweet and label
    df = df[columns_to_keep].copy()
    
    # Remove duplicates
    print(f"\n🔄 Removing duplicates...")
    df = df.drop_duplicates(subset=['isi_tweet']).reset_index(drop=True)
    print(f"   ✓ After removing duplicates: {len(df):,} rows")
    
    # Remove empty texts
    df = df[df['isi_tweet'].str.len() > 0].reset_index(drop=True)
    print(f"   ✓ After removing empty texts: {len(df):,} rows")
    
    # Statistics
    print(f"\n📊 Final class distribution:")
    for label, name in [(-1, "Negative"), (0, "Neutral"), (1, "Positive")]:
        count = len(df[df['label'] == label])
        pct = 100 * count / len(df)
        print(f"   {name:10s}: {count:6,d} ({pct:5.1f}%)")
    
    # Save
    print(f"\n💾 Saving cleaned data: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"   ✓ Saved successfully")
    
    # Show sample
    print(f"\n📋 Sample rows:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"\n   Row {idx+1}:")
        print(f"      isi_tweet: {row['isi_tweet'][:80]}...")
        print(f"      label: {row['label']}")
    
    print(f"\n✅ Data cleaning completed!")
    print(f"   Final dataset: {len(df):,} samples")


if __name__ == "__main__":
    clean_data()
