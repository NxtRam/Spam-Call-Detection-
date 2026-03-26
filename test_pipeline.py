import sys, time
from real_time_classifier import RealTimeClassifier, CallMonitor
from call_prevention import CallPrevention

classifier = RealTimeClassifier()
# Ensure model override doesn't crash
if hasattr(classifier.model, 'n_jobs'): classifier.model.n_jobs = 1

monitor = CallMonitor(classifier)
prev = CallPrevention(block_threshold=0.85, persistent_block=True)

test_texts = ["hello i am calling from the bank", "you have won a lottery", "please click on this link"]

for text in test_texts:
    print(f"\nProcessing: {text}")
    l, c = classifier.classify(text)
    print(f"Single fragment: {l} ({c:.2f})")
    
    cl, cc, persistent = monitor.add_fragment(text)
    print(f"Rolling buffer: {cl} ({cc:.2f}) Persistent: {persistent}")
    print(f"Stats: {monitor.get_call_summary()}")
    
    b, reason = prev.should_block(l, c, persistent, False)
    if b:
        print(f"BLOCKED: {reason}")
    else:
        print("Not blocked")

print("Done")
