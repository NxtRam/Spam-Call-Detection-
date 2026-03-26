import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asr_engine import ASREngine
from real_time_classifier import RealTimeClassifier, StreamMonitor, CallMonitor

app = FastAPI(title="Telephony ASR WebSocket Server")

# Global instances
asr_engine = ASREngine()
classifier = RealTimeClassifier()

@app.get("/")
async def root():
    return {"status": "online", "message": "Telephony ASR Server"}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    
    # Per-connection monitors
    monitor = StreamMonitor(classifier, threshold=0.92)
    call_monitor = CallMonitor(classifier)
    
    # Task to run the periodic spam check
    async def monitor_task():
        try:
            while True:
                await asyncio.sleep(5)
                alert = monitor.check_for_spam()
                if alert:
                    print(alert)
                    await websocket.send_json({"type": "alert", "content": alert})
        except Exception as e:
            print(f"Monitor task stopped: {e}")

    # Start the monitor in the background
    asyncio.create_task(monitor_task())

    try:
        while True:
            # Expecting raw binary audio chunks (PCM 16-bit 8000Hz)
            data = await websocket.receive_bytes()
            
            # Process with Vosk
            result_json = asr_engine.process_chunk(data)
            result = json.loads(result_json)
            
            if 'text' in result and result['text']:
                text = result['text']
                monitor.update_transcript(text)
                
                label, confidence = classifier.classify(text)
                await websocket.send_json({
                    "type": "final",
                    "text": text,
                    "label": label,
                    "confidence": confidence
                })
                print(f"[Final] ({label} - {confidence:.2f}) {text}")
                
                # Update call monitor
                cum_label, cum_conf, is_persistent = call_monitor.add_fragment(text)
                if is_persistent:
                    await websocket.send_json({
                        "type": "persistent_warning",
                        "content": f"PERSISTENT SPAM (Confidence: {cum_conf:.2f})"
                    })
                
            elif 'partial' in result and result['partial']:
                await websocket.send_json({
                    "type": "partial",
                    "text": result['partial']
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        call_monitor.reset()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
