import os
import time
from faster_whisper import WhisperModel

AUDIO_DIR = "../dataset/test.audio"
OUTPUT_DIR = "Results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "faster_whisper_outputs.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Faster-Whisper model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"   
)

results = []

for audio_file in sorted(os.listdir(AUDIO_DIR)):
    if not audio_file.endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    print(f"Processing: {audio_file}")

    start_time = time.time()
    segments, info = model.transcribe(audio_path)
    end_time = time.time()

    predicted_text = " ".join([seg.text.strip() for seg in segments])
    inference_time = round(end_time - start_time, 3)

    results.append(
        f"File: {audio_file}\n"
        f"Prediction: {predicted_text}\n"
        f"Inference Time: {inference_time} sec\n"
        f"{'-'*60}\n"
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

print("\n Faster-Whisper testing completed")
print(f" Results saved at: {OUTPUT_FILE}")
