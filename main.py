import json
import time
import threading
import sys
import numpy as np
from audio_capture import AudioCapture
from asr_engine import ASREngine
from real_time_classifier import RealTimeClassifier, StreamMonitor, CallMonitor, SpeechAnalyzer

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║       ON-DEVICE SCAM CALL DETECTION & PREVENTION        ║
║              Real-Time Speech Analysis                   ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def print_call_summary(call_monitor, classifier):
    """Prints a summary report of the call analysis."""
    summary = call_monitor.get_call_summary()
    spam_pct = summary['spam_percentage']

    if spam_pct >= 70:
        verdict = f"{RED}{BOLD}⛔ HIGH RISK - LIKELY SCAM CALL{RESET}"
    elif spam_pct >= 40:
        verdict = f"{YELLOW}{BOLD}⚠️  MEDIUM RISK - SUSPICIOUS CALL{RESET}"
    elif spam_pct >= 15:
        verdict = f"{YELLOW}🔍 LOW RISK - SOME SUSPICIOUS CONTENT{RESET}"
    else:
        verdict = f"{GREEN}✅ SAFE - NORMAL CALL{RESET}"

    print(f"\n{CYAN}{'═' * 58}{RESET}")
    print(f"{CYAN}{BOLD}  CALL ANALYSIS SUMMARY{RESET}")
    print(f"{CYAN}{'═' * 58}{RESET}")
    print(f"  Total speech fragments analyzed: {summary['total_fragments']}")
    print(f"  Spam fragments detected:         {summary['spam_fragments']}")
    print(f"  Spam percentage:                 {spam_pct:.1f}%")
    print(f"  Persistent spam detected:        {'YES ⚠️' if summary['persistent_spam'] else 'No'}")
    print(f"\n  {BOLD}VERDICT:{RESET} {verdict}")
    print(f"{CYAN}{'═' * 58}{RESET}\n")

def main():
    print_banner()
    print("Initializing system components...")

    # Initialize ASR Engine (downloads model if missing)
    engine = ASREngine(model_path="model")
    print(f"  {GREEN}✓{RESET} ASR Engine (Vosk) loaded")

    # Initialize Classifier and Monitors
    classifier = RealTimeClassifier()
    monitor = StreamMonitor(classifier, threshold=0.70)
    call_monitor = CallMonitor(classifier)
    speech_analyzer = SpeechAnalyzer()
    print(f"  {GREEN}✓{RESET} Spam classifier loaded")
    print(f"  {GREEN}✓{RESET} Stream & call monitors ready")

    # Parse device index from CLI if provided
    device_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # Initialize Audio Capture
    capture = AudioCapture(rate=16000, chunk=1024, input_device_index=device_idx)
    print(f"  {GREEN}✓{RESET} Audio capture initialized")

    def monitor_loop():
        """Background thread to check accumulated transcript for spam every 5 seconds."""
        while True:
            time.sleep(5)
            alert = monitor.check_for_spam()
            if alert:
                print(f"\n{alert}")

    monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitoring_thread.start()

    print(f"\n{CYAN}{'─' * 58}{RESET}")
    print(f"  {BOLD}Listening...{RESET} Speak into your microphone.")
    print(f"  Press {BOLD}Ctrl+C{RESET} to stop and view call summary.")
    print(f"{CYAN}{'─' * 58}{RESET}\n")

    try:
        capture.start_stream()
        last_result_time = time.time()
        warning_shown = False

        for audio_chunk in capture.get_audio_stream():
            # Calculate current energy (RMS) for activity display
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_data.astype(float)**2))

            result_json = engine.process_chunk(audio_chunk)
            result = json.loads(result_json)

            if 'text' in result and result['text']:
                now = time.time()
                duration = now - last_result_time
                last_result_time = now

                # Normalize ASR output: fix phonetic mis-transcriptions
                # e.g. "oh tee pee" → "otp", "kay why see" → "kyc"
                raw_text = result['text']
                text = ASREngine.normalize_text(raw_text)
                # Feed to stream monitor
                monitor.update_transcript(text)

                # Analyze for Robocall patterns (speed, filler words)
                is_robo, wps, f_dens = speech_analyzer.analyze(text, duration)
                robo_flag = f" {RED}[ROBOCALL]{RESET}" if is_robo else ""

                # Classify the text
                label, confidence = classifier.classify(text)
                risk = classifier.get_risk_level(confidence)

                if label == "Spam":
                    status_tag = f"[{RED}SPAM{RESET}]"
                else:
                    status_tag = f"[{GREEN}HAM {RESET}]"

                print(f"\r{status_tag} ({confidence:.2f}) {risk}{robo_flag} {text}")

                # Spam alert with prevention advice
                if label == "Spam" and confidence >= 0.70:
                    print(f"   └─ {RED}{BOLD}⚠️  ALERT: Potential scam detected!{RESET}")
                    if confidence >= 0.85 and not warning_shown:
                        print(f"   └─ {RED}{BOLD}🛑 RECOMMENDATION: Disconnect this call immediately!{RESET}")
                        print(f"   └─ {YELLOW}Do NOT share OTP, bank details, or personal information.{RESET}")
                        warning_shown = True

                if is_robo:
                    print(f"   └─ {YELLOW}Audio Analysis: {wps:.1f} words/sec | Fillers: {f_dens:.2f}{RESET}")

                # Rolling buffer analysis for persistent spam
                cum_label, cum_conf, is_persistent = call_monitor.add_fragment(text)
                if is_persistent:
                    print(f"\n{RED}{BOLD}🛑 PERSISTENT SCAM DETECTED! (Rolling Confidence: {cum_conf:.2f}){RESET}")
                    print(f"{RED}   ⚠️  This call shows consistent scam patterns. HANG UP NOW!{RESET}")
                    print(f"{YELLOW}   📞 Report this number to cybercrime.gov.in or call 1930{RESET}\n")

            elif 'partial' in result and result['partial']:
                print(f"[RMS: {energy:5.1f}] [Partial] {result['partial']}", end="\r")
            else:
                print(f"[RMS: {energy:5.1f}] Listening...", end="\r")

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopping capture...{RESET}")
    finally:
        capture.stop_stream()
        print_call_summary(call_monitor, classifier)

if __name__ == "__main__":
    main()
