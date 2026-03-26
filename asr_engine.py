from vosk import Model, KaldiRecognizer
import os
import re
import sys
import zipfile
import requests

# Phonetic / misheard word corrections for scam-related vocabulary.
# Vosk's small model often transcribes acronyms and Indian terms phonetically.
PHONETIC_CORRECTIONS = {
    # OTP variations
    'oh tee pee': 'otp', 'oh t p': 'otp', 'o t p': 'otp',
    'oh tp': 'otp', 'ot pee': 'otp', 'otb': 'otp',
    'or to be': 'otp', 'owed he be': 'otp',

    # KYC variations
    'kay why see': 'kyc', 'k y c': 'kyc', 'key wycee': 'kyc',
    'k why see': 'kyc', 'kay y c': 'kyc', 'key y see': 'kyc',
    'k y see': 'kyc', 'k wise he': 'kyc',

    # CBI variations
    'see bee eye': 'cbi', 'c b i': 'cbi', 'sea be i': 'cbi',
    'see be i': 'cbi', 'cbi i': 'cbi',

    # SIM variations
    'sim card': 'sim card', 'seem card': 'sim card', 'seem': 'sim',

    # FIR variations
    'f i r': 'fir', 'eff i r': 'fir', 'f i are': 'fir',

    # UPI variations
    'you pee eye': 'upi', 'u p i': 'upi', 'you p i': 'upi',
    'you pie': 'upi',

    # PAN card
    'pan card': 'pan card', 'panned card': 'pan card',
    'p a n': 'pan', 'pee a n': 'pan',

    # Aadhaar variations
    'are hard': 'aadhaar', 'odd har': 'aadhaar', 'other': 'aadhaar',
    'odd hard': 'aadhaar', 'our card': 'aadhaar', 'a door': 'aadhaar',
    'at art': 'aadhaar', 'odd are': 'aadhaar',

    # TRAI variations
    'try': 'trai', 'tray': 'trai', 't r a i': 'trai',

    # CVV variations
    'see vee vee': 'cvv', 'c v v': 'cvv', 'see we we': 'cvv',

    # RBI variations
    'are be eye': 'rbi', 'r b i': 'rbi', 'are bee i': 'rbi',

    # NIA variations
    'n i a': 'nia', 'and i a': 'nia',

    # IFSC
    'i f s c': 'ifsc', 'eye f s c': 'ifsc',

    # KBC
    'k b c': 'kbc', 'kay bee see': 'kbc', 'k bc': 'kbc',

    # Common scam phrases Vosk might garble
    'digital arrest': 'digital arrest',
    'money laundering': 'money laundering',
    'arrest warrant': 'arrest warrant',
    'bank account': 'bank account',
    'credit card': 'credit card',
    'debit card': 'debit card',
    'gift card': 'gift card',
    'scratch card': 'scratch card',
    'lucky draw': 'lucky draw',
    'press one': 'press one',
    'press nine': 'press nine',

    # Hinglish corrections
    'band ho jayega': 'band ho jayega',
    'block ho gaya': 'block ho gaya',
    'grift toss': 'grifters',
    'onee': 'won',
}


class ASREngine:
    def __init__(self, model_path="model"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            self._download_small_model()

        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, 16000)

    def _download_small_model(self):
        print(f"Model not found at {self.model_path}. Downloading a small English model...")
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = "model.zip"

        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            extracted_dir = zip_ref.namelist()[0].split('/')[0]
            os.rename(extracted_dir, self.model_path)

        os.remove(zip_path)
        print("Model downloaded and extracted.")

    @staticmethod
    def normalize_text(text):
        """
        Post-processes ASR output to correct common misheard scam keywords.
        Replaces phonetic transcriptions with the correct terms.
        """
        if not text:
            return text

        normalized = text.lower().strip()

        # Apply phonetic corrections (longest match first to avoid partial replacements)
        sorted_corrections = sorted(PHONETIC_CORRECTIONS.keys(), key=len, reverse=True)
        for phonetic, correct in ((p, PHONETIC_CORRECTIONS[p]) for p in sorted_corrections):
            if phonetic in normalized:
                normalized = normalized.replace(phonetic, correct)

        return normalized

    def process_chunk(self, data):
        if self.rec.AcceptWaveform(data):
            return self.rec.Result()
        else:
            return self.rec.PartialResult()


if __name__ == "__main__":
    # Test normalization
    engine = ASREngine()
    print("ASR Engine ready.\n")

    test_inputs = [
        "please share your oh tee pee for kay why see verification",
        "this is see bee eye calling about your are hard card",
        "your you pee eye pin is needed to process the refund",
        "see vee vee number of your debit card please share",
        "are be eye alert your bank account will be frozen",
        "hello how are you doing today",
    ]

    print("--- Text Normalization Tests ---")
    for text in test_inputs:
        corrected = ASREngine.normalize_text(text)
        print(f"  IN:  {text}")
        print(f"  OUT: {corrected}\n")
