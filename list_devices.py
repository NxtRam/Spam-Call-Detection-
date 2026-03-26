import pyaudio

def list_devices():
    p = pyaudio.PyAudio()
    print("Available Audio Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Index {i}: {info['name']} (Inputs: {info['maxInputChannels']}, Rate: {info['defaultSampleRate']})")
    p.terminate()

if __name__ == "__main__":
    list_devices()
