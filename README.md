# Sarcasm Detector

[![CI](https://github.com/YOUR_USERNAME/sarcasm-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/sarcasm-detector/actions)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-Backened%2Fsarcasm--model-yellow)](https://huggingface.co/Backened/sarcasm-model)
[![Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/sarcasm-detector)
[![API](https://img.shields.io/badge/API-Render.com-green)](https://sarcasm-detector.onrender.com/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> Fine-tuned **DistilBERT** for sarcasm detection in English text.  
> Trained on **120,000+ samples** from Reddit, Twitter, and news headlines.  
> Served via **FastAPI** · Deployed on **Render** · Demo on **HuggingFace Spaces**.

---

## Live Demo

**Try it now → [huggingface.co/spaces/YOUR_USERNAME/sarcasm-detector](https://huggingface.co/spaces/YOUR_USERNAME/sarcasm-detector)**

```
POST https://sarcasm-detector.onrender.com/predict

{
  "text": "Oh great, another Monday. Just what I needed."
}

→ { "label": "Sarcastic", "confidence": 0.93, "latency_ms": 42 }
```

---

## Results

| Metric    | Score  |
|-----------|--------|
| F1        | 0.872  |
| Accuracy  | 85.3%  |
| Precision | 0.883  |
| Recall    | 0.862  |

**Per-domain breakdown:**

| Domain    | Samples | F1    |
|-----------|---------|-------|
| Reddit    | ~20k    | ~0.88 |
| Twitter   | ~8k     | ~0.84 |
| Headlines | ~5k     | ~0.91 |

---

## Architecture

```
Raw text (Reddit · Twitter · Headlines)
        │
        ▼
  Minimal cleaning       ← URLs + mentions only
  (BERT-safe)            ← NO lemmatization, NO stopword removal
        │
        ▼
  HuggingFace Tokenizer  ← distilbert-base-uncased, max_length=128
        │
        ▼
  DistilBERT fine-tune   ← 2 epochs · lr=2e-5 · batch=32 · fp16
  (120k samples)         ← EarlyStopping patience=3
        │
        ▼
  Best checkpoint        ← Saved at val loss minimum (step ~2500)
        │
        ▼
  FastAPI  ──────────────── /predict  /batch  /health
  Render.com hosting
        │
        ▼
  Gradio demo ────────────  HuggingFace Spaces
```

---

## Quickstart

### Option 1 — Use the API directly

```bash
curl -X POST https://sarcasm-detector.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Oh sure, everything is totally fine."}'
```

### Option 2 — Python SDK

```python
import requests

response = requests.post(
    "https://sarcasm-detector.onrender.com/predict",
    json={"text": "Oh great, another update that breaks everything."}
)
print(response.json())
# {'label': 'Sarcastic', 'confidence': 0.91, 'is_uncertain': False, 'latency_ms': 38.4}
```

### Option 3 — Run locally

```bash
git clone https://github.com/YOUR_USERNAME/sarcasm-detector
cd sarcasm-detector

cp .env.example .env
# Edit .env → set HF_REPO_ID=Backened/sarcasm-model

pip install -r requirements.txt
uvicorn api.main:app --reload
# → open http://localhost:8000/docs
```

### Option 4 — Docker

```bash
docker build -t sarcasm-detector .
docker run -p 8000:8000 sarcasm-detector
```

---

## API Reference

### `POST /predict`

| Field       | Type   | Default | Description                              |
|-------------|--------|---------|------------------------------------------|
| `text`      | string | —       | Input text (3–512 chars)                 |
| `threshold` | float  | 0.65    | Confidence below this → returns Uncertain|

**Response:**
```json
{
  "text": "Oh great, another Monday.",
  "label": "Sarcastic",
  "confidence": 0.93,
  "is_uncertain": false,
  "latency_ms": 41.2,
  "model_version": "distilbert-sarcasm-v1"
}
```

### `POST /batch`

Send up to **100 texts** in one request.

```json
{ "texts": ["text1", "text2", "text3"], "threshold": 0.65 }
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

Full interactive docs → [`/docs`](https://sarcasm-detector.onrender.com/docs)

---

## Project Structure

```
sarcasm-detector/
├── api/
│   ├── main.py          # FastAPI app — /predict /batch /health
│   └── model.py         # Singleton model loader from HF Hub
├── demo/
│   └── app.py           # Gradio demo (HuggingFace Spaces)
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI — runs on every push
├── Dockerfile           # Containerized API
├── requirements.txt     # Pinned dependencies
├── .env.example         # Environment variable template
└── README.md
```

---

## Training Details

| Parameter         | Value                      |
|-------------------|----------------------------|
| Base model        | `distilbert-base-uncased`  |
| Training samples  | ~120,000                   |
| Epochs            | 2 (EarlyStopping at 1.91)  |
| Learning rate     | 2e-5                       |
| Batch size        | 32                         |
| Max length        | 128 tokens                 |
| Optimizer         | AdamW + cosine schedule    |
| Mixed precision   | fp16                       |
| Best checkpoint   | Step ~2500 (lowest val loss)|

**Datasets used:**
- [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- [Reddit Sarcasm](https://www.kaggle.com/datasets/danofer/sarcasm)
- [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhilmittal/tweets-with-sarcasm-and-irony)

Training notebook → [`/kaggle`](https://www.kaggle.com/YOUR_KAGGLE_USERNAME)

---

## Key Engineering Decisions

**Why minimal text cleaning?**  
Sarcasm signals live in stopwords and punctuation — "Oh great", "just what I needed", "sure...". Standard NLP cleaning (lemmatization, stopword removal) destroys these markers before the model can learn them. DistilBERT's tokenizer handles normalization internally.

**Why split before upsampling?**  
Upsampling before splitting causes data leakage — duplicate rows appear in both train and test sets, inflating reported F1. The correct pipeline: split on original data → upsample training fold only.

**Why EarlyStopping at patience=3?**  
Validation loss hit its minimum at step ~2500 (epoch 0.87). The model began mild overfitting after that — train loss kept dropping while val loss plateaued. EarlyStopping automatically restored the best checkpoint.

---

## License

MIT © YOUR_NAME

---

## Contact

**YOUR_NAME** · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [HuggingFace](https://huggingface.co/Backened)
