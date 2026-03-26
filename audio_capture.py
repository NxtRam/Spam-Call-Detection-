import pyaudio

class AudioCapture:
    def __init__(self, rate=16000, chunk=1024, input_device_index=None):
        self.rate = rate
        self.chunk = chunk
        self.input_device_index = input_device_index
        self.format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self):
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk
            )
            msg = f"* Capture started at {self.rate}Hz"
            if self.input_device_index is not None:
                msg += f" (Device Index: {self.input_device_index})"
            print(msg + "...")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not start audio stream: {e}")
            if "err='-50'" in str(e) or "Unknown Error" in str(e):
                print("HINT: Try a different device index or check microphone permissions.")
            raise e

    def get_audio_stream(self):
        if not self.stream:
            self.start_stream()
        
        try:
            while True:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                yield data
        except KeyboardInterrupt:
            self.stop_stream()

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("* Capture stopped.")

if __name__ == "__main__":
    # Test capture
    ac = AudioCapture()
    try:
        for chunk in ac.get_audio_stream():
            print(f"Captured {len(chunk)} bytes", end="\r")
    except KeyboardInterrupt:
        pass
