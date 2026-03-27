# SentimentScope Submission

IMDB binary sentiment classification using custom transformer models.

## Contents

| File | Description |
|------|-------------|
| `SentimentScope_complete.ipynb` | From-scratch DemoGPT (~9.5M params, ~87% test accuracy). Saves `demogpt_imdb.pt` locally after full retrain and includes a batch inference cell. |
| `SentimentScope_bert_like.ipynb` | BERT architecture + pretrained weights (~109M params, ~93.6% test accuracy). Saves `demogpt_bertlike.pt` locally during training. |
| `SentimentScope_Research_Report.md` | Full research report documenting the optimization journey |
| `hyperparameter_optimization_log.txt` | Raw logs from Optuna (Part A) and manual experiments (Part B) |
| `aclImdb/` | Extracted dataset (train/test, pos/neg) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

This repository already includes `aclImdb/` for direct reproducibility.

## Running

Both notebooks should be run from this directory so that relative paths to `aclImdb/` resolve correctly.

- **`SentimentScope_complete.ipynb`** — runs in ~13 minutes on an RTX 4080 Laptop GPU (8 epochs + 5-epoch full retrain). Run cells in order through training; the checkpoint and inference cells expect the working directory to be this folder (same as `aclImdb/` paths).
- **`SentimentScope_bert_like.ipynb`** — runs in ~30 minutes (3 epochs fine-tuning). BERT pretrained weights are downloaded automatically by HuggingFace's `transformers` library on first run.

A CUDA-capable GPU is strongly recommended. Training on CPU is possible but will be significantly slower.

## Rubric Checklist

- **Train a model with >75% test accuracy**  
  Covered in `SentimentScope_complete.ipynb` (from-scratch DemoGPT, ~87% test) and `SentimentScope_bert_like.ipynb` (~93%+ test).

- **Create a report summarizing project results with key takeaways**  
  Covered in `SentimentScope_Research_Report.md`, including final results summary and multiple ranked insights/takeaways.
