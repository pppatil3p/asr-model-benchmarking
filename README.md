# ASR Model Benchmarking for Customer Support Applications

This repository presents a comparative evaluation of three state-of-the-art Automatic Speech Recognition (ASR) models—**Whisper (Small)**, **Faster-Whisper**, and **Wav2Vec 2.0**—with the objective of identifying the most suitable model for deployment in noisy, real-world customer support environments.

The project focuses on benchmarking transcription accuracy, inference efficiency, and deployment feasibility using a subset of the Mozilla Common Voice dataset.

---

## 📌 Objectives

- Evaluate multiple ASR models under a consistent experimental setup
- Compare models based on **Word Error Rate (WER)**, inference time, and memory usage
- Analyze trade-offs between accuracy, robustness, and efficiency
- Recommend an ASR model suitable for real-world customer support systems

---

## 📁 Repository Structure

asr-model-benchmarking/
├── ARS_test/ # Evaluation scripts and results
│ ├── whisper_test.py
│ ├── faster_whisper_test.py
│ ├── wav2vec2_test.py
│ ├── wer_eval.py
│ └── Results/
│ ├── whisper_outputs.txt
│ ├── faster_whisper_outputs.txt
│ ├── wav2vec2_outputs.txt
│ └── wer_results.txt
│
├── dataset/
│ └── transcript/ # Ground-truth transcriptions
│
├── Research-Report/
│ ├── Technical Report.docx
│ └── Executive Summary.docx
│
├── Research_Papers/ # Reference papers
├── .gitignore
├── requirements.txt
└── README.md


*Note: Audio files are excluded to keep the repository lightweight.*

---

## 📊 Evaluation Metrics

- **Word Error Rate (WER)** – Primary metric for transcription accuracy
- **Approximate Accuracy** – Derived as (1 − WER)
- **Inference Time** – Measured during model execution
- **Memory Usage** – Observed qualitatively
- **Deployment Suitability** – Qualitative assessment

---

## 📈 Results Summary

| Model | Avg WER ↓ | Approx. Accuracy |
|------|----------|------------------|
| Whisper (Small) | 0.314 | 68.6% |
| Faster-Whisper | 0.306 | 69.4% |
| Wav2Vec 2.0 | **0.190** | **81.0%** |

*WER values were computed on 24 audio samples; one sample was excluded due to missing ground-truth transcription.*

---

## 🧠 Key Findings

- **Wav2Vec 2.0** achieved the lowest WER on the selected dataset, indicating strong performance on relatively clean and moderately noisy speech.
- **Whisper (Small)** demonstrated high robustness to accents and background noise but incurred higher inference latency and memory usage.
- **Faster-Whisper** provided the best balance between accuracy, robustness, and efficiency, making it more suitable for real-time and scalable deployment.

---

## ✅ Final Recommendation

Although Wav2Vec 2.0 achieved higher accuracy on this dataset, **Faster-Whisper is recommended for real-world customer support applications** due to its consistent robustness, faster inference, and lower resource requirements under unpredictable acoustic conditions.

---

 ## Author
**Prathmesh Patil**  
