import os
from jiwer import wer

TRANSCRIPT_DIR = "../dataset/transcript"
RESULTS_DIR = "Results"

MODEL_OUTPUTS = {
    "Whisper": "whisper_outputs.txt",
    "Faster-Whisper": "faster_whisper_outputs.txt",
    "Wav2Vec2": "wav2vec2_outputs.txt"
}

OUTPUT_FILE = os.path.join(RESULTS_DIR, "wer_results.txt")

def load_ground_truth(audio_filename):
    base_name = os.path.splitext(audio_filename)[0]

    candidates = [
        base_name + ".txt",
        "test" + base_name + ".txt"
    ]

    for cand in candidates:
        path = os.path.join(TRANSCRIPT_DIR, cand)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip().lower()

    return None  

def extract_predictions(output_file):
    predictions = {}
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read().split("-" * 60)

    for block in content:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        file_name = lines[0].replace("File: ", "").strip()
        pred_text = lines[1].replace("Prediction: ", "").strip().lower()
        predictions[file_name] = pred_text

    return predictions

summary = []

for model_name, output_name in MODEL_OUTPUTS.items():
    print(f"\nEvaluating {model_name}...")
    preds = extract_predictions(os.path.join(RESULTS_DIR, output_name))

    wers = []
    skipped = 0

    for audio_file, pred_text in preds.items():
        gt_text = load_ground_truth(audio_file)

        if gt_text is None:
            skipped += 1
            continue

        wers.append(wer(gt_text, pred_text))

    avg_wer = round(sum(wers) / len(wers), 3)

    summary.append(
        f"{model_name} Average WER: {avg_wer} "
        f"(evaluated on {len(wers)} files, skipped {skipped})\n"
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(summary)

print("\n WER evaluation completed")
print(f" Results saved at: {OUTPUT_FILE}")
