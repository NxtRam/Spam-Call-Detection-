import eventlet
eventlet.monkey_patch()

import os
import time
import json
import uuid
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

from audio_capture import AudioCapture
from asr_engine import ASREngine
from real_time_classifier import RealTimeClassifier, StreamMonitor, CallMonitor, SpeechAnalyzer
from call_prevention import ScamBlocklist, ScamReportGenerator, CallPrevention

app = Flask(__name__)
# Enable CORS for socketio
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state to track active call
active_call = {
    'is_running': False,
    'capture': None,
    'thread': None,
    'session_id': None
}

# Initialize AI models once at startup to avoid delay
print("Initializing AI models...")
engine = ASREngine(model_path="model")
classifier = RealTimeClassifier()
blocklist = ScamBlocklist()
report_gen = ScamReportGenerator()

@app.route('/')
def index():
    return render_template('index.html')

def call_analysis_task():
    """Background task that runs the audio loop and emits events via WebSocket."""
    global active_call
    
    # Initialize fresh monitors for this call
    monitor = StreamMonitor(classifier, threshold=0.55)
    call_monitor = CallMonitor(classifier)
    speech_analyzer = SpeechAnalyzer()
    prevention = CallPrevention(block_threshold=0.85, persistent_block=True)
    
    session_id = str(uuid.uuid4())
    start_time = time.time()
    full_transcript = ""
    scam_fragments_log = []
    auto_blocked = False
    
    try:
        capture = AudioCapture(rate=16000, chunk=1024)
        active_call['capture'] = capture
        capture.start_stream()
        
        last_result_time = time.time()
        
        # Tell frontend we are listening
        socketio.emit('call_status', {'status': 'listening'})
        
        for audio_chunk in capture.get_audio_stream():
            if not active_call['is_running']:
                break
                
            # Calculate RMS energy for UI audio visualizer
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = float(np.sqrt(np.mean(audio_data.astype(float)**2)))
            socketio.emit('audio_level', {'energy': energy})
            
            result_json = engine.process_chunk(audio_chunk)
            result = json.loads(result_json)
            
            if 'text' in result and result['text']:
                now = time.time()
                duration = now - last_result_time
                last_result_time = now
                
                raw_text = result['text']
                text = ASREngine.normalize_text(raw_text)
                full_transcript += " " + text
                
                monitor.update_transcript(text)
                is_robo, wps, f_dens = speech_analyzer.analyze(text, duration)
                label, confidence = classifier.classify(text)
                risk = classifier.get_risk_level(confidence)
                
                # Emit the finalized text fragment with its analysis
                socketio.emit('transcript_final', {
                    'text': text,
                    'label': label,
                    'confidence': float(confidence),
                    'risk_level': risk,
                    'is_robo': is_robo
                })
                
                # Log scam fragments to DB
                if label == "Spam" and confidence >= 0.50:
                    scam_fragments_log.append({
                        'text': text, 'confidence': confidence, 'risk_level': risk, 'is_robocall': is_robo
                    })
                    blocklist.log_scam_fragment(session_id, text, confidence, risk, is_robo)
                
                # Emit a high-level alert separately if potent scam
                if label == "Spam" and confidence >= 0.55:
                    socketio.emit('scam_alert', {
                        'message': 'Potential scam detected!',
                        'confidence': float(confidence),
                        'critical': confidence >= 0.80
                    })
                
                # Rolling buffer analysis for persistent spam UI updates
                cum_label, cum_conf, is_persistent = call_monitor.add_fragment(text)
                
                # Emit overall call stats to update the UI dashboard
                summary = call_monitor.get_call_summary()
                socketio.emit('call_stats', {
                    'spam_percentage': summary['spam_percentage'],
                    'total_fragments': summary['total_fragments'],
                    'spam_fragments': summary['spam_fragments'],
                    'is_persistent': is_persistent,
                    'rolling_confidence': float(cum_conf)
                })
                
                # ─── PREVENTION CHECK ───
                should_block, block_reason = prevention.should_block(
                    label, confidence, is_persistent, is_robo
                )
                if should_block and not auto_blocked:
                    auto_blocked = True
                    socketio.emit('auto_block', {
                        'reason': block_reason
                    })
                    
                    # Log db, gen report
                    end_time = time.time()
                    try:
                        active_call['capture'].stop_stream()
                    except: pass
                    
                    # Generate report and add to blocklist
                    blocklist.log_call_session(session_id, start_time, end_time, summary, full_transcript.strip(), True)
                    blocklist.add_to_blocklist("Unknown", "Scam Caller", f"Auto-detected: {summary['spam_percentage']:.0f}% spam", "CRITICAL")
                    report_path = report_gen.generate_report(session_id, start_time, end_time, summary, full_transcript.strip(), scam_fragments_log, True)
                    
                    os.system('say "Scam call detected. Disconnecting."')
                    
                    # Tell frontend the call was dropped and where report is
                    socketio.emit('call_dropped', {'report': report_path})
                    active_call['is_running'] = False
                    break
                    
            elif 'partial' in result and result['partial']:
                # Emit partial text for real-time typing effect
                socketio.emit('transcript_partial', {
                    'text': result['partial']
                })
                
    except Exception as e:
        print(f"Error in audio capture: {e}")
    finally:
        active_call['is_running'] = False
        try:
            if active_call['capture']:
                active_call['capture'].stop_stream()
        except: pass
        socketio.emit('call_status', {'status': 'ended'})
        print("Analysis thread exiting.")

@socketio.on('start_call')
def handle_start_call():
    global active_call
    if active_call['is_running']:
        return
        
    print("UI triggered start_call")
    active_call['is_running'] = True
    active_call['thread'] = threading.Thread(target=call_analysis_task)
    active_call['thread'].daemon = True
    active_call['thread'].start()

@socketio.on('end_call')
def handle_end_call():
    global active_call
    print("UI triggered end_call")
    active_call['is_running'] = False
    
    if active_call['capture']:
        try:
            active_call['capture'].stop_stream()
        except: pass

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("=" * 60)
    print(" STARTING WEB DASHBOARD ON http://127.0.0.1:5000")
    print("=" * 60)
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
