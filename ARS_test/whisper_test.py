import whisper
import os
import time

AUDIO_DIR = "../dataset/test.audio"
OUTPUT_DIR = "Results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "whisper_outputs.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")  

results = []

for audio_file in sorted(os.listdir(AUDIO_DIR)):
    if not audio_file.endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file)

    print(f"Processing: {audio_file}")

    start_time = time.time()
    result = model.transcribe(audio_path)
    end_time = time.time()

    inference_time = round(end_time - start_time, 3)
    predicted_text = result["text"].strip()

    results.append(
        f"File: {audio_file}\n"
        f"Prediction: {predicted_text}\n"
        f"Inference Time: {inference_time} sec\n"
        f"{'-'*60}\n"
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

print("\n Whisper testing completed")
print(f" Results saved at: {OUTPUT_FILE}")
