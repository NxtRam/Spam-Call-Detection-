import json
import time
import threading
import sys
import uuid
import numpy as np
from audio_capture import AudioCapture
from asr_engine import ASREngine
from real_time_classifier import RealTimeClassifier, StreamMonitor, CallMonitor, SpeechAnalyzer
from call_prevention import ScamBlocklist, ScamReportGenerator, CallPrevention

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"
BG_RED = "\033[41m"

def print_banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║       ON-DEVICE SCAM CALL DETECTION & PREVENTION        ║
║              Real-Time Speech Analysis                   ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def print_block_alert(reason):
    """Displays a prominent auto-block alert."""
    print(f"\n{BG_RED}{BOLD}")
    print("  ╔════════════════════════════════════════════════════╗")
    print("  ║           🛑 CALL AUTO-BLOCKED 🛑                ║")
    print("  ╠════════════════════════════════════════════════════╣")
    print(f"  ║  Reason: {reason:<43}║")
    print("  ╠════════════════════════════════════════════════════╣")
    print("  ║  ⚠️  DO NOT share any personal information        ║")
    print("  ║  ⚠️  DO NOT share OTP, PIN, or bank details       ║")
    print("  ║  ⚠️  HANG UP the call IMMEDIATELY                 ║")
    print("  ║                                                    ║")
    print("  ║  📞 Report: cybercrime.gov.in or call 1930         ║")
    print("  ╚════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")

def print_call_summary(call_monitor, classifier, blocklist, session_id,
                       start_time, end_time, full_transcript,
                       scam_fragments_log, auto_blocked, report_gen, prevention):
    """Prints call summary and generates report if scam detected."""
    summary = call_monitor.get_call_summary()
    spam_pct = summary['spam_percentage']

    if spam_pct >= 70:
        verdict = f"{RED}{BOLD}⛔ HIGH RISK - LIKELY SCAM CALL{RESET}"
        verdict_text = "HIGH RISK - LIKELY SCAM"
    elif spam_pct >= 40:
        verdict = f"{YELLOW}{BOLD}⚠️  MEDIUM RISK - SUSPICIOUS CALL{RESET}"
        verdict_text = "MEDIUM RISK - SUSPICIOUS"
    elif spam_pct >= 15:
        verdict = f"{YELLOW}🔍 LOW RISK - SOME SUSPICIOUS CONTENT{RESET}"
        verdict_text = "LOW RISK"
    else:
        verdict = f"{GREEN}✅ SAFE - NORMAL CALL{RESET}"
        verdict_text = "SAFE"

    summary['verdict'] = verdict_text
    summary['max_confidence'] = max([f['confidence'] for f in scam_fragments_log], default=0)

    print(f"\n{CYAN}{'═' * 58}{RESET}")
    print(f"{CYAN}{BOLD}  CALL ANALYSIS SUMMARY{RESET}")
    print(f"{CYAN}{'═' * 58}{RESET}")
    print(f"  Session ID:                      {session_id[:8]}...")
    print(f"  Total speech fragments analyzed:  {summary['total_fragments']}")
    print(f"  Spam fragments detected:          {summary['spam_fragments']}")
    print(f"  Spam percentage:                  {spam_pct:.1f}%")
    print(f"  Max spam confidence:              {summary['max_confidence']:.2f}")
    print(f"  Persistent spam detected:         {'YES ⚠️' if summary['persistent_spam'] else 'No'}")
    print(f"  Call auto-blocked:                {'YES 🛑' if auto_blocked else 'No'}")

    print(f"\n  {BOLD}VERDICT:{RESET} {verdict}")

    # -- PREVENTION ACTIONS --
    if spam_pct >= 30 or auto_blocked:
        print(f"\n{CYAN}{'─' * 58}{RESET}")
        print(f"{CYAN}{BOLD}  PREVENTION ACTIONS TAKEN{RESET}")
        print(f"{CYAN}{'─' * 58}{RESET}")

        # 1. Log to database
        blocklist.log_call_session(session_id, start_time, end_time,
                                   summary, full_transcript, auto_blocked)
        print(f"  {GREEN}✓{RESET} Call session logged to database")

        # 2. Add to blocklist if high risk
        if spam_pct >= 50:
            risk = "CRITICAL" if spam_pct >= 70 else "HIGH"
            blocklist.add_to_blocklist("Unknown", "Scam Caller",
                                       f"Auto-detected: {spam_pct:.0f}% spam", risk)
            print(f"  {GREEN}✓{RESET} Caller added to blocklist ({risk})")

        # 3. Generate evidence report
        report_path = report_gen.generate_report(
            session_id, start_time, end_time, summary,
            full_transcript, scam_fragments_log, auto_blocked
        )
        print(f"  {GREEN}✓{RESET} Scam evidence report saved: {report_path}")

        # 4. Show reporting info
        print(f"\n  {YELLOW}{BOLD}📋 WHAT TO DO NEXT:{RESET}")
        print(f"  {YELLOW}  1. Report at https://cybercrime.gov.in{RESET}")
        print(f"  {YELLOW}  2. Call Cyber Crime Helpline: 1930{RESET}")
        print(f"  {YELLOW}  3. If bank details shared, call your bank NOW{RESET}")
        print(f"  {YELLOW}  4. Evidence report saved at: {report_path}{RESET}")
    else:
        # Still log safe calls
        blocklist.log_call_session(session_id, start_time, end_time,
                                   summary, full_transcript, False)

    # Show stats
    stats = blocklist.get_stats()
    print(f"\n{CYAN}{'─' * 58}{RESET}")
    print(f"{CYAN}{BOLD}  PROTECTION STATS{RESET}")
    print(f"{CYAN}{'─' * 58}{RESET}")
    print(f"  Numbers in blocklist:     {stats['blocked_numbers']}")
    print(f"  Total calls analyzed:     {stats['total_calls_analyzed']}")
    print(f"  Calls auto-blocked:       {stats['calls_auto_blocked']}")
    print(f"  Scam fragments caught:    {stats['scam_fragments_detected']}")
    print(f"{CYAN}{'═' * 58}{RESET}\n")


def main():
    print_banner()
    print("Initializing system components...")

    # Initialize ASR Engine
    engine = ASREngine(model_path="model")
    print(f"  {GREEN}✓{RESET} ASR Engine (Vosk) loaded")

    # Initialize Classifier and Monitors
    classifier = RealTimeClassifier()
    monitor = StreamMonitor(classifier, threshold=0.55)
    call_monitor = CallMonitor(classifier)
    speech_analyzer = SpeechAnalyzer()
    print(f"  {GREEN}✓{RESET} Spam classifier loaded")
    print(f"  {GREEN}✓{RESET} Stream & call monitors ready")

    # Initialize Prevention System
    blocklist = ScamBlocklist()
    report_gen = ScamReportGenerator()
    prevention = CallPrevention(block_threshold=0.85, persistent_block=True)
    print(f"  {GREEN}✓{RESET} Prevention engine armed")
    print(f"  {GREEN}✓{RESET} Scam blocklist database loaded")

    # Session tracking
    session_id = str(uuid.uuid4())
    start_time = time.time()
    full_transcript = ""
    scam_fragments_log = []
    auto_blocked = False

    # Parse device index from CLI if provided
    device_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    capture = AudioCapture(rate=16000, chunk=1024, input_device_index=device_idx)
    print(f"  {GREEN}✓{RESET} Audio capture initialized")

    # Show protection stats
    stats = blocklist.get_stats()
    if stats['blocked_numbers'] > 0:
        print(f"\n  {MAGENTA}📊 {stats['blocked_numbers']} numbers blocked | "
              f"{stats['total_calls_analyzed']} calls analyzed so far{RESET}")

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
    print(f"  {MAGENTA}🛡️  Auto-block is ARMED for scam prevention{RESET}")
    print(f"{CYAN}{'─' * 58}{RESET}\n")

    try:
        capture.start_stream()
        last_result_time = time.time()
        warning_shown = False

        for audio_chunk in capture.get_audio_stream():
            # Calculate current energy (RMS)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_data.astype(float)**2))

            result_json = engine.process_chunk(audio_chunk)
            result = json.loads(result_json)

            if 'text' in result and result['text']:
                now = time.time()
                duration = now - last_result_time
                last_result_time = now

                # Normalize ASR output
                raw_text = result['text']
                text = ASREngine.normalize_text(raw_text)
                full_transcript += " " + text

                # Feed to stream monitor
                monitor.update_transcript(text)

                # Analyze for Robocall patterns
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

                # Log scam fragments
                if label == "Spam" and confidence >= 0.50:
                    scam_fragments_log.append({
                        'text': text,
                        'confidence': confidence,
                        'risk_level': risk,
                        'is_robocall': is_robo
                    })
                    blocklist.log_scam_fragment(session_id, text, confidence, risk, is_robo)

                # Spam alert with prevention advice
                if label == "Spam" and confidence >= 0.55:
                    print(f"   └─ {RED}{BOLD}⚠️  ALERT: Potential scam detected!{RESET}")
                    if confidence >= 0.80 and not warning_shown:
                        print(f"   └─ {RED}{BOLD}🛑 RECOMMENDATION: Disconnect this call immediately!{RESET}")
                        print(f"   └─ {YELLOW}Do NOT share OTP, bank details, or personal information.{RESET}")
                        warning_shown = True

                if is_robo:
                    print(f"   └─ {YELLOW}Audio Analysis: {wps:.1f} words/sec | Fillers: {f_dens:.2f}{RESET}")

                # Rolling buffer analysis for persistent spam
                cum_label, cum_conf, is_persistent = call_monitor.add_fragment(text)

                # ─── PREVENTION CHECK ───
                should_block, block_reason = prevention.should_block(
                    label, confidence, is_persistent, is_robo
                )
                if should_block and not auto_blocked:
                    auto_blocked = True
                    print_block_alert(block_reason)
                    print(f"{RED}{BOLD}  📵 AUTO-DISCONNECTING CALL...{RESET}\n")
                    
                    # Instead of breaking, we forcibly end the session to 'hang up' immediately
                    end_time = time.time()
                    capture.stop_stream()
                    print_call_summary(call_monitor, classifier, blocklist, session_id,
                                      start_time, end_time, full_transcript.strip(),
                                      scam_fragments_log, auto_blocked, report_gen, prevention)
                    print(f"{RED}{BOLD}\nCall disconnected by Scam Prevention System.{RESET}")
                    
                    # Auditory feedback using macOS 'say' command
                    import os
                    os.system('say "Scam call detected. Disconnecting."')
                    sys.exit(0)

                if is_persistent and not auto_blocked:
                    print(f"\n{RED}{BOLD}🛑 PERSISTENT SCAM DETECTED! (Confidence: {cum_conf:.2f}){RESET}")
                    print(f"{RED}   ⚠️  This call shows consistent scam patterns. HANG UP NOW!{RESET}")
                    print(f"{YELLOW}   📞 Report: cybercrime.gov.in or call 1930{RESET}\n")

            elif 'partial' in result and result['partial']:
                blocked_tag = f" {BG_RED}{BOLD}[BLOCKED]{RESET}" if auto_blocked else ""
                print(f"[RMS: {energy:5.1f}]{blocked_tag} [Partial] {result['partial']}", end="\r")
            else:
                blocked_tag = f" {BG_RED}{BOLD}[BLOCKED]{RESET}" if auto_blocked else ""
                print(f"[RMS: {energy:5.1f}]{blocked_tag} Listening...", end="\r")

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopping capture...{RESET}")
    finally:
        end_time = time.time()
        capture.stop_stream()
        print_call_summary(call_monitor, classifier, blocklist, session_id,
                          start_time, end_time, full_transcript.strip(),
                          scam_fragments_log, auto_blocked, report_gen, prevention)

if __name__ == "__main__":
    main()
