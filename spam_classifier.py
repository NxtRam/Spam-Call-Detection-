import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_FILE = 'spam_model.joblib'
DATASET_FILE = 'scam_call_dataset.csv'

class SpamClassifier:
    def __init__(self):
        self.pipeline = None
        if os.path.exists(MODEL_FILE):
            self.load_model()

    def build_pipeline(self):
        """Creates the ML pipeline using TfidfVectorizer and Random Forest."""
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=5000,
                sublinear_tf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, csv_path=DATASET_FILE, text_col='text', label_col='label'):
        """Trains the model on a CSV dataset."""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df[label_col] = df[label_col].str.strip()

        X = df[text_col]
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Random Forest model...")
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        predictions = self.pipeline.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, predictions))

        self.save_model()

    def save_model(self):
        joblib.dump(self.pipeline, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

    def load_model(self):
        self.pipeline = joblib.load(MODEL_FILE)
        print("Model loaded from disk.")

    def predict(self, text, threshold=0.70):
        """Predicts 'Spam' or 'Ham' using probability threshold."""
        if self.pipeline is None:
            return "Model not trained"

        if len(text.split()) < 2:
            return "Ham"

        prob = self.get_spam_probability(text)
        return "Spam" if prob >= threshold else "Ham"

    def get_spam_probability(self, text):
        """Returns the probability of the text being 'Spam'."""
        if self.pipeline is None:
            return 0.0

        if len(text.split()) < 2:
            return 0.0

        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_

        spam_idx = list(classes).index('Spam')
        return probs[spam_idx]


if __name__ == "__main__":
    classifier = SpamClassifier()
    classifier.train(DATASET_FILE)

    # Test Predictions
    test_cases = [
        "Your account is under digital arrest. A narcotics case has been registered.",
        "Hey can you send me the presentation file before the meeting?",
        "You have won a lottery prize. Call now to claim your reward.",
    ]

    print("\n--- Test Predictions ---")
    for text in test_cases:
        result = classifier.predict(text)
        prob = classifier.get_spam_probability(text)
        print(f"[{result}] ({prob:.2f}) {text[:60]}...")
