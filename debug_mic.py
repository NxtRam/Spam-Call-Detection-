import pyaudio
import numpy as np

def test_mic():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        print("Listening for 5 seconds...")
        for _ in range(10):
            data = stream.read(8000, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_data.astype(float)**2))
            print(f"Audio energy: {energy:.2f}")
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_mic()
