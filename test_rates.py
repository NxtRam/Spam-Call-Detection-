import pyaudio

def test_native_rate(index=2, rate=48000):
    p = pyaudio.PyAudio()
    try:
        print(f"Attempting to open device {index} at {rate}Hz...")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(rate),
            input=True,
            input_device_index=index,
            frames_per_buffer=1024
        )
        print("Success! Reading 10 chunks...")
        for _ in range(10):
            stream.read(1024)
        print("Test passed.")
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Failed at {rate}Hz: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    test_native_rate(idx, 48000)
    test_native_rate(idx, 16000)
