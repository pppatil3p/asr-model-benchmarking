import os
import time
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

AUDIO_DIR = "../dataset/test.audio"
OUTPUT_DIR = "Results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wav2vec2_outputs.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Wav2Vec 2.0 model...")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model.eval()

results = []

for audio_file in sorted(os.listdir(AUDIO_DIR)):
    if not audio_file.endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    print(f"Processing: {audio_file}")

    speech, sr = librosa.load(audio_path, sr=16000)

    start_time = time.time()

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(predicted_ids)[0]

    end_time = time.time()
    inference_time = round(end_time - start_time, 3)

    results.append(
        f"File: {audio_file}\n"
        f"Prediction: {predicted_text}\n"
        f"Inference Time: {inference_time} sec\n"
        f"{'-'*60}\n"
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

print("\n Wav2Vec 2.0 testing completed")
print(f" Results saved at: {OUTPUT_FILE}")
