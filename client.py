import asyncio
import websockets
import json
from audio_capture import AudioCapture

async def stream_audio():
    uri = "ws://localhost:8000/ws/audio"
    capture = AudioCapture(rate=8000)
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Streaming microphone audio...")
            
            # Start the audio stream
            capture.start_stream()
            
            # Task to receive messages from the server
            async def receive_messages():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data["type"] == "final":
                            print(f"\n[Server Final] ({data['label']}) {data['text']}")
                        elif data["type"] == "alert":
                            print(f"\n[ALERT] {data['content']}")
                        elif data["type"] == "partial":
                            print(f"\r[Server Partial] {data['text']}", end="", flush=True)
                except websockets.ConnectionClosed:
                    print("\nConnection closed by server.")

            # Start receiver task
            receiver = asyncio.create_task(receive_messages())

            # Stream audio chunks
            for chunk in capture.get_audio_stream():
                await websocket.send(chunk)
                await asyncio.sleep(0.01) # Small sleep to yield to receiver

    except Exception as e:
        print(f"Error: {e}")
    finally:
        capture.stop_stream()

if __name__ == "__main__":
    try:
        asyncio.run(stream_audio())
    except KeyboardInterrupt:
        pass
