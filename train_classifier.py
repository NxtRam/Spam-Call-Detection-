import pandas as pd
import joblib
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
MODEL_FILE = 'rf_model.joblib'
DATASET_FILE = 'scam_call_dataset.csv'

def train_spam_model(csv_path=DATASET_FILE, text_col='text', label_col='label'):
    """Train a scam call detection model on a call-transcript dataset."""
    print("=" * 60)
    print("  SCAM CALL DETECTION - MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset '{csv_path}' not found!")
        print("Please ensure scam_call_dataset.csv exists in the project directory.")
        return

    # ----- Load Data -----
    df = pd.read_csv(csv_path)
    # Strip whitespace from labels
    df[label_col] = df[label_col].str.strip()
    print(f"\nDataset: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"  Spam: {len(df[df[label_col] == 'Spam'])}")
    print(f"  Ham:  {len(df[df[label_col] == 'Ham'])}")

    X = df[text_col]
    y = df[label_col]

    # ----- Train/Test Split -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ----- Feature Engineering: TF-IDF -----
    print("\nExtracting TF-IDF features (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=5000,
        sublinear_tf=True  # Apply log normalization for better performance
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_vec.shape}")

    # ----- Model Training: Random Forest -----
    print("\nTraining Random Forest Classifier (200 trees, balanced)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)

    # ----- Evaluation -----
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Ham', 'Spam'])
    print(f"  {'':>8} Predicted")
    print(f"  {'':>8} {'Ham':>6} {'Spam':>6}")
    print(f"  {'Ham':>8} {cm[0][0]:>6} {cm[0][1]:>6}")
    print(f"  {'Spam':>8} {cm[1][0]:>6} {cm[1][1]:>6}")

    # ----- Cross Validation -----
    print("\nRunning 5-fold Cross Validation...")
    X_all_vec = vectorizer.transform(X)
    cv_scores = cross_val_score(model, X_all_vec, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ----- Save Components -----
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved: {VECTORIZER_FILE}")
    print(f"Saved: {MODEL_FILE}")

    # ----- Save Vocabulary for Reference -----
    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    with open('vocabulary.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved: vocabulary.json ({len(vocab)} terms)")

    # ----- Save Training Report -----
    with open('training_report.txt', 'w') as f:
        f.write("SCAM CALL DETECTION - TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {csv_path}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Spam: {len(df[df[label_col] == 'Spam'])} | Ham: {len(df[df[label_col] == 'Ham'])}\n")
        f.write(f"Train: {len(X_train)} | Test: {len(X_test)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
    print(f"Saved: training_report.txt")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_spam_model()
