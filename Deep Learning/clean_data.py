"""
Clean data - fix anomalies and update for binary classification
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")

print("=" * 70)
print("CLEANING DATA")
print("=" * 70)

# Load
df = pd.read_csv(CSV_PATH)
print(f"\n📊 Before cleaning:")
print(f"   Total rows: {len(df):,}")
print(f"   Unique labels: {sorted(df['label'].unique())}")

# Remove anomalies (not -1, 0, or 1)
valid_labels = {-1.0, 0.0, 1.0}
df_clean = df[df['label'].isin(valid_labels)].copy()

# Remove neutral (0)
df_clean = df_clean[df_clean['label'] != 0.0].copy()

print(f"\n📊 After cleaning:")
print(f"   Total rows: {len(df_clean):,}")
print(f"   Unique labels: {sorted(df_clean['label'].unique())}")
print(f"\n   Class distribution:")
for label in sorted(df_clean['label'].unique()):
    name = "Negative" if label == -1.0 else "Positive"
    count = len(df_clean[df_clean['label'] == label])
    pct = 100 * count / len(df_clean)
    print(f"      {name:10s}: {count:6,d} ({pct:5.1f}%)")

# Save
df_clean.to_csv(CSV_PATH, index=False)
print(f"\n💾 Saved: {CSV_PATH}")
