"""
Remove neutral sentiment data (label = 0)
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")

print("=" * 80)
print("REMOVING NEUTRAL LABELS")
print("=" * 80)

# Load data
print(f"\n📄 Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"   Total rows before: {len(df):,}")

# Print class distribution before
print(f"\n📊 Class distribution before:")
for label, name in [(-1.0, "Negative"), (0.0, "Neutral"), (1.0, "Positive")]:
    count = len(df[df['label'] == label])
    pct = 100 * count / len(df)
    print(f"   {name:10s}: {count:6,d} ({pct:5.1f}%)")

# Remove neutral
df_filtered = df[df['label'] != 0.0].copy()
print(f"\n✂️  Removing neutral labels...")
print(f"   Total rows after: {len(df_filtered):,}")
print(f"   Rows removed: {len(df) - len(df_filtered):,}")

# Print class distribution after
print(f"\n📊 Class distribution after:")
for label, name in [(-1.0, "Negative"), (1.0, "Positive")]:
    count = len(df_filtered[df_filtered['label'] == label])
    pct = 100 * count / len(df_filtered)
    print(f"   {name:10s}: {count:6,d} ({pct:5.1f}%)")

# Save
print(f"\n💾 Saving...")
df_filtered.to_csv(CSV_PATH, index=False)
print(f"   ✓ Saved: {CSV_PATH}")

print(f"\n{'='*80}")
print(f"✅ DONE!")
print(f"{'='*80}")
