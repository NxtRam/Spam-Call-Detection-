import joblib
import os
import re
import time

VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
MODEL_FILE = 'rf_model.joblib'

# ---- SCAM KEYWORD DATABASE ----
# These are substrings that indicate scam content, even in garbled ASR output.
# Using short substrings to catch partial/garbled words from Vosk.
SCAM_SUBSTRINGS = [
    # Core scam action words
    'arrest', 'warrant', 'seize', 'prosecut', 'lawsuit',
    'block', 'frozen', 'suspend', 'deactiv', 'disconnect', 'terminat',
    'illegal', 'fraud', 'launder', 'narcotic', 'smuggl',
    # Urgency / pressure words
    'urgent', 'immediately', 'right now', 'within two hour',
    'do not hang up', 'do not disconnect', 'stay on the line',
    'final notice', 'last warning', 'legal action',
    # Financial theft words
    'otp', 'cvv', 'pin number', 'upi pin', 'card number',
    'share your', 'verify your', 'send money', 'transfer fund',
    'pay fine', 'pay fee', 'registration fee', 'processing fee',
    'bank detail', 'account number', 'refund',
    # Identity words
    'kyc', 'pan card', 'aadhaar', 'aadhar', 'adhaar',
    # Authority impersonation
    'cbi', 'police department', 'crime branch', 'customs',
    'rbi', 'reserve bank', 'trai', 'enforcement',
    'income tax', 'court case', 'supreme court',
    'narcotics department', 'electricity department',
    # Prize / lottery scams
    'lottery', 'winner', 'won prize', 'won a gift', 'lucky draw',
    'congratulat', 'claim now', 'claim your', 'gift card',
    'scratch card', 'cash prize', 'kbc',
    # Investment scams
    'guaranteed return', 'double your money', 'no risk',
    'invest now', 'crypto', 'trading platform',
    # Telecom scams
    'sim block', 'sim band', 'sim card kyc',
    'number will be block', 'service terminated',
    # Common scam phrases
    'press one', 'press nine', 'press 1', 'press 9',
    'digital arrest', 'money launder',
    'password', 'passwor', 'passwo',
    'scam', 'spam',
]

# High-confidence scam phrases (if detected, very likely scam)
HIGH_CONFIDENCE_PHRASES = [
    'digital arrest', 'arrest warrant', 'money laundering',
    'share otp', 'share your otp', 'otp share',
    'kyc pending', 'kyc update', 'kyc expire',
    'account block', 'account frozen', 'account band',
    'sim block', 'sim band', 'sim deactivat',
    'won lottery', 'won prize', 'kbc lottery',
    'narcotics', 'illegal parcel', 'seized package',
    'upi pin', 'cvv number', 'card number share',
    'do not hang up', 'stay on video call',
]


class RealTimeClassifier:
    def __init__(self, threshold=0.55):
        self.vectorizer = None
        self.model = None
        self.threshold = threshold
        self.load_components()

    def load_components(self):
        """Loads vectorizer and model separately for optimized inference."""
        if os.path.exists(VECTORIZER_FILE) and os.path.exists(MODEL_FILE):
            self.vectorizer = joblib.load(VECTORIZER_FILE)
            self.model = joblib.load(MODEL_FILE)
            return True
        return False

    def _keyword_score(self, text):
        """
        Scans text for scam keywords/substrings.
        Returns a score between 0.0 and 1.0 based on keyword density.
        Works even with garbled ASR output by using substring matching.
        """
        text_lower = text.lower()
        
        # Check high-confidence phrases first
        high_hits = sum(1 for phrase in HIGH_CONFIDENCE_PHRASES if phrase in text_lower)
        if high_hits > 0:
            return min(0.85 + (high_hits * 0.05), 1.0)

        # Count regular scam substring matches
        hits = sum(1 for kw in SCAM_SUBSTRINGS if kw in text_lower)

        if hits == 0:
            return 0.0
        elif hits == 1:
            return 0.35
        elif hits == 2:
            return 0.55
        elif hits == 3:
            return 0.70
        elif hits <= 5:
            return 0.85
        else:
            return 0.95

    def classify(self, text):
        """
        Hybrid classifier: combines ML model + keyword detection.
        Returns: (label, confidence_score)
        """
        if not text or len(text.strip()) < 2:
            return "Ham", 0.0

        # 1. Keyword-based score (works on garbled text)
        keyword_score = self._keyword_score(text)

        # 2. ML model score (works best on clean text)
        ml_score = 0.0
        if self.vectorizer and self.model and len(text.split()) >= 2:
            vectorized_text = self.vectorizer.transform([text])
            probs = self.model.predict_proba(vectorized_text)[0]
            classes = self.model.classes_
            spam_idx = list(classes).index('Spam')
            ml_score = probs[spam_idx]

        # 3. Combine: take the higher of the two scores
        # This ensures keyword hits work even when ML model fails on garbled text
        confidence = max(keyword_score, ml_score)

        # Small boost when both agree
        if keyword_score > 0.3 and ml_score > 0.3:
            confidence = min(confidence + 0.10, 1.0)

        label = "Spam" if confidence >= self.threshold else "Ham"
        return label, confidence

    def get_risk_level(self, confidence):
        """Maps confidence score to a human-readable risk level."""
        if confidence >= 0.90:
            return "🔴 CRITICAL"
        elif confidence >= 0.75:
            return "🟠 HIGH"
        elif confidence >= 0.50:
            return "🟡 MEDIUM"
        elif confidence >= 0.30:
            return "🟢 LOW"
        else:
            return "⚪ SAFE"


class StreamMonitor:
    """Monitors the incoming ASR stream and alerts on accumulated spam."""
    def __init__(self, classifier, threshold=0.55):
        self.classifier = classifier
        self.threshold = threshold
        self.last_checked_index = 0
        self.full_transcript = ""

    def update_transcript(self, new_text):
        self.full_transcript += " " + new_text
        self.full_transcript = self.full_transcript.strip()

    def check_for_spam(self):
        new_content = self.full_transcript[self.last_checked_index:].strip()
        if not new_content:
            return None

        label, confidence = self.classifier.classify(new_content)
        self.last_checked_index = len(self.full_transcript)

        if confidence >= self.threshold:
            risk = self.classifier.get_risk_level(confidence)
            return f"🚨 SPAM ALERT {risk} (Conf: {confidence:.2f}) - '{new_content[:80]}...'"

        return None


class CallMonitor:
    """Tracks classification results over a rolling time window to detect persistent spam."""
    def __init__(self, classifier, window_seconds=30, persistent_threshold=0.50):
        self.classifier = classifier
        self.window_seconds = window_seconds
        self.persistent_threshold = persistent_threshold

        self.buffer = []
        self.consecutive_spam_count = 0
        self.persistent_flag = False
        self.total_fragments = 0
        self.spam_fragments = 0

    def reset(self):
        self.buffer = []
        self.consecutive_spam_count = 0
        self.persistent_flag = False
        print("CallMonitor buffer reset.")

    def add_fragment(self, text):
        """
        Appends new text, cleans window, and classifies accumulated transcript.
        Returns: (label, confidence, is_persistent_spam)
        """
        now = time.time()
        self.buffer.append((now, text))
        self.total_fragments += 1

        # Rolling window
        self.buffer = [f for f in self.buffer if now - f[0] <= self.window_seconds]
        accumulated_text = " ".join([f[1] for f in self.buffer])

        label, confidence = self.classifier.classify(accumulated_text)

        if label == "Spam":
            self.spam_fragments += 1

        if confidence >= self.persistent_threshold:
            self.consecutive_spam_count += 1
        else:
            self.consecutive_spam_count = 0

        if self.consecutive_spam_count >= 3:
            self.persistent_flag = True

        return label, confidence, self.persistent_flag

    def get_call_summary(self):
        spam_pct = (self.spam_fragments / self.total_fragments * 100) if self.total_fragments > 0 else 0
        return {
            'total_fragments': self.total_fragments,
            'spam_fragments': self.spam_fragments,
            'spam_percentage': spam_pct,
            'persistent_spam': self.persistent_flag
        }


class SpeechAnalyzer:
    """Analyzes speech patterns to detect robocall characteristics."""
    def __init__(self, velocity_threshold=3.5, filler_threshold=0.0):
        self.velocity_threshold = velocity_threshold
        self.fillers = {'um', 'uh', 'err', 'ah', 'like', 'hmmm', 'hmm'}

    def analyze(self, text, duration_seconds):
        if duration_seconds <= 0:
            return False, 0.0, 0.0

        words = text.lower().split()
        word_count = len(words)
        wps = word_count / duration_seconds

        filler_hits = [w for w in words if w in self.fillers]
        filler_density = len(filler_hits) / word_count if word_count > 0 else 0.0

        is_robocall = (wps >= self.velocity_threshold) and (filler_density <= self.filler_threshold)
        return is_robocall, wps, filler_density


if __name__ == "__main__":
    rtc = RealTimeClassifier()

    test_cases = [
        # Should detect as SPAM (even garbled versions)
        ("your otp is needed for kyc verification", "Scam: OTP + KYC"),
        ("share your password and account number", "Scam: password + account"),
        ("this is cbi calling arrest warrant issued", "Scam: CBI + arrest"),
        ("congratulations you won lottery prize", "Scam: lottery"),
        ("scam call detected password", "Scam: scam + password"),
        ("digital arrest money laundering case", "Scam: digital arrest"),
        ("sim block ho jayega kyc pending", "Scam: sim block + kyc"),
        ("press one to verify your bank detail", "Scam: press one + bank"),
        # Should detect as HAM
        ("hello how are you doing today", "Normal: greeting"),
        ("can you pick up milk from the store", "Normal: daily task"),
        ("the meeting is at three pm tomorrow", "Normal: work"),
        ("happy birthday wish you a great year", "Normal: wishes"),
        ("lets go for dinner tonight", "Normal: casual"),
    ]

    print("=" * 65)
    print("  HYBRID SCAM DETECTOR - TEST RESULTS")
    print("  (ML Model + Keyword Detection)")
    print("=" * 65)
    for text, desc in test_cases:
        label, conf = rtc.classify(text)
        risk = rtc.get_risk_level(conf)
        tag = "\033[91mSPAM\033[0m" if label == "Spam" else "\033[92mHAM \033[0m"
        print(f"\n  [{tag}] ({conf:.2f}) {risk}")
        print(f"    {desc}: \"{text}\"")
