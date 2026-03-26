"""
Download the UCI SMS Spam Collection dataset and prepare it for training.
Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
Same dataset as: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
"""
import os
import io
import zipfile
import csv
import requests
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
OUTPUT_CSV = "sms_spam_full.csv"
COMBINED_CSV = "scam_call_dataset.csv"

def download_uci_dataset():
    """Downloads and extracts the UCI SMS Spam Collection."""
    print("=" * 60)
    print("  DOWNLOADING UCI SMS SPAM COLLECTION")
    print("=" * 60)

    print(f"\nFetching from: {DATASET_URL}")
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    print(f"Downloaded {len(response.content)} bytes")

    # Extract the TSV file from the zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        print(f"ZIP contents: {zf.namelist()}")
        # The main file is 'SMSSpamCollection'
        with zf.open('SMSSpamCollection') as f:
            content = f.read().decode('utf-8')

    # Parse the tab-separated data
    rows = []
    for line in content.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) == 2:
            label = parts[0].strip()
            text = parts[1].strip()
            # Normalize labels: 'ham' -> 'Ham', 'spam' -> 'Spam'
            label = 'Spam' if label.lower() == 'spam' else 'Ham'
            rows.append({'text': text, 'label': label})

    df_uci = pd.DataFrame(rows)
    print(f"\nUCI Dataset loaded: {len(df_uci)} samples")
    print(f"  Spam: {len(df_uci[df_uci['label'] == 'Spam'])}")
    print(f"  Ham:  {len(df_uci[df_uci['label'] == 'Ham'])}")

    # Save standalone UCI dataset
    df_uci.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved UCI dataset to: {OUTPUT_CSV}")

    # Also load our custom scam call dataset and merge
    custom_csv = "scam_call_dataset.csv"
    if os.path.exists(custom_csv):
        df_custom = pd.read_csv(custom_csv)
        df_custom['label'] = df_custom['label'].str.strip()
        print(f"\nCustom dataset loaded: {len(df_custom)} samples")

        # Combine both datasets
        df_combined = pd.concat([df_uci, df_custom], ignore_index=True)
        # Remove duplicates
        df_combined = df_combined.drop_duplicates(subset='text', keep='first')
        print(f"\nCombined dataset: {len(df_combined)} samples")
        print(f"  Spam: {len(df_combined[df_combined['label'] == 'Spam'])}")
        print(f"  Ham:  {len(df_combined[df_combined['label'] == 'Ham'])}")

        df_combined.to_csv(COMBINED_CSV, index=False)
        print(f"Saved combined dataset to: {COMBINED_CSV}")
    else:
        # If no custom dataset, just use UCI as the main dataset
        df_uci.to_csv(COMBINED_CSV, index=False)
        print(f"Saved as: {COMBINED_CSV}")

    print("\n" + "=" * 60)
    print("  DATASET READY FOR TRAINING")
    print(f"  Run: python train_classifier.py")
    print("=" * 60)

if __name__ == "__main__":
    download_uci_dataset()
