import joblib
import m2cgen as m2c
import json
import os
import sys

# Increase recursion limit for larger models (200 trees with deep decision paths)
sys.setrecursionlimit(50000)

MODEL_FILE = 'spam_model.joblib'

def export_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Please run spam_classifier.py first.")
        return

    # Load the pipeline
    print(f"Loading pipeline from {MODEL_FILE}...")
    pipeline = joblib.load(MODEL_FILE)
    
    # Extract the vectorizer and classifier
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    
    # 1. Export the Classifier using m2cgen (to Java for Android or C for iOS)
    print("Generating Java code for Random Forest...")
    java_code = m2c.export_to_java(classifier)
    with open("SpamModel.java", "w") as f:
        f.write(java_code)
    print("Exported SpamModel.java")

    print("Generating C code for Random Forest...")
    c_code = m2c.export_to_c(classifier)
    with open("spam_model.c", "w") as f:
        f.write(c_code)
    print("Exported spam_model.c")

    # 2. Export the TF-IDF Metadata
    # On mobile, you will need to compute: TF(word) * IDF(word)
    # Convert numpy types to standard python types for JSON serialization
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    idf_weights = vectorizer.idf_.tolist()
    
    tf_idf_metadata = {
        "vocabulary": vocabulary,
        "idf": idf_weights,
        "ngram_range": vectorizer.ngram_range,
        "stop_words": list(vectorizer.get_stop_words()) if vectorizer.get_stop_words() else []
    }
    
    with open("tfidf_metadata.json", "w") as f:
        json.dump(tf_idf_metadata, f)
    print("Exported tfidf_metadata.json (includes vocabulary and IDF weights)")

    print("\n--- Mobile Export Summary ---")
    print("1. SpamModel.java / spam_model.c: Pure code implementation of the RF model.")
    print("2. tfidf_metadata.json: Use this for feature extraction (TF * IDF) on mobile.")
    print("3. Latency: These pure implementations will run in <10ms for a single prediction.")

if __name__ == "__main__":
    export_model()
