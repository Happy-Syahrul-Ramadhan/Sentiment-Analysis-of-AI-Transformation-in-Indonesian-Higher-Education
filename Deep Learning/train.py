"""
train.py — Pipeline Training Sentiment Classification (ML Models)
==================================================================
Jalankan:
    python train.py

Alur:
    1. Data preprocessing (lowercase, hapus punctuation, dll)
    2. Load data dari combined_training_data.csv
    3. Split train/val/test
    4. Vectorisasi teks dengan TF-IDF
    5. Training multiple models (Naive Bayes, Random Forest, SVM)
    6. Evaluasi dan pilih best model
    7. Simpan model & visualisasi results
"""

import os
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)

# ──────────────────────────────────────────────
# CONSTANTS & PATHS
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

CSV_PATH = os.path.join(DATA_DIR, "combined_training_data.csv")

# Create directories
for d in [MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# Hyperparameters
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# TF-IDF Vectorizer params
TFIDF_MAX_FEATURES = 5000
TFIDF_MAX_DF = 0.8
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_data(csv_path):
    """Load dan prepare data."""
    print("=" * 80)
    print("LOADING COMBINED DATA")
    print("=" * 80)
    
    print(f"\n📄 Loading: {csv_path}")
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"   ✓ Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    df = df[['isi_tweet', 'label']].dropna()
    print(f"   Total rows: {len(df):,}")
    
    print(f"\n📊 Class distribution:")
    for label, name in [(-1, "Negative"), (0, "Neutral"), (1, "Positive")]:
        count = len(df[df['label'] == label])
        pct = 100 * count / len(df)
        print(f"   {name:10s}: {count:6,d} ({pct:5.1f}%)")
    
    return df['isi_tweet'].values, df['label'].values


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train multiple models dan select best."""
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    # Vectorize
    print(f"\n📝 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   ✓ Feature matrix shape: {X_train_vec.shape}")
    
    # Models
    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        ),
        "SVM": SVC(kernel='rbf', C=1.0, random_state=RANDOM_SEED, verbose=1),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n▶ Training {name}...")
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start_time
        
        y_val_pred = model.predict(X_val_vec)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"   ✓ Training completed in {train_time:.2f}s")
        print(f"   ✓ Validation Accuracy: {acc:.4f}")
        
        results[name] = {
            'model': model,
            'vectorizer': vectorizer,
            'val_acc': acc,
            'train_time': train_time
        }
    
    # Select best
    best_name = max(results.keys(), key=lambda k: results[k]['val_acc'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['val_acc']
    
    print(f"\n📊 Training Summary:")
    for name in results.keys():
        time_val = results[name]['train_time']
        acc_val = results[name]['val_acc']
        print(f"   {name:15s}: Acc={acc_val:.4f} | Time={time_val:.2f}s")
    
    print(f"\n🏆 Best Model: {best_name} (Val Acc: {best_acc:.4f})")
    
    # Test
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*80}")
    
    y_test_pred = best_model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted', zero_division=0
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=['Negative', 'Positive'],
        zero_division=0
    ))
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "sentiment_classifier.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    
    print(f"\n💾 Saving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Model: {model_path}")
    print(f"   ✓ Vectorizer: {vectorizer_path}")
    
    # Plot confusion matrix
    print(f"\n📊 Creating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_test_pred, labels=[-1, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(PLOT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    print(f"   ✓ Saved: {cm_path}")
    plt.close()
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': [best_name],
        'Val_Accuracy': [best_acc],
        'Test_Accuracy': [test_acc],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
    })
    
    metrics_path = os.path.join(PLOT_DIR, 'training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ✓ Saved metrics: {metrics_path}")
    
    return best_model, vectorizer, y_test_pred, y_test


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    """Main training pipeline."""
    print("\n🚀 SENTIMENT ANALYSIS - TRAINING WITH COMBINED DATA")
    print("=" * 80)
    
    # Load data
    texts, labels = load_data(CSV_PATH)
    
    # Split: 80% train, 10% val, 10% test with stratification
    print(f"\n📊 Splitting data (train:val:test = 80:10:10)...")
    
    # First: Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.20,
        random_state=RANDOM_SEED
    )
    
    # Second: Split train into train (90%) and val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.111,  # ~10% of original 80%
        random_state=RANDOM_SEED
    )
    
    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")
    
    # Train
    model, vectorizer, y_pred, y_true = train_and_evaluate(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved in:")
    print(f"   Models: {MODEL_DIR}/")
    print(f"   Plots:  {PLOT_DIR}/")


if __name__ == "__main__":
    main()
