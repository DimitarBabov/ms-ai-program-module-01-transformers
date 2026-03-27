# SentimentScope Submission

IMDB binary sentiment classification using custom transformer models.

## Contents

| File | Description |
|------|-------------|
| `SentimentScope_complete.ipynb` | From-scratch DemoGPT (~9.5M params, ~87% test accuracy). Saves `demogpt_imdb.pt` after the full 25k retrain; next cell loads it and runs inference on a random test batch. |
| `demogpt_imdb.pt` | Model weights written by the complete notebook (omit from git if size is an issue; re-run training to regenerate). |
| `SentimentScope_bert_like.ipynb` | BERT architecture + pretrained weights (~109M params, ~93.6% test accuracy). Writes `demogpt_bertlike.pt` during training. |
| `demogpt_bertlike.pt` | BERT-like notebook checkpoint (optional in git; re-run the notebook to regenerate). |
| `SentimentScope_Research_Report.md` | Full research report documenting the optimization journey |
| `hyperparameter_optimization_log.txt` | Raw logs from Optuna (Part A) and manual experiments (Part B) |
| `aclImdb_v1.tar.gz` | IMDB dataset archive |
| `aclImdb/` | Extracted dataset (train/test, pos/neg) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

If `aclImdb/` is missing, extract the dataset:

```bash
tar -xzf aclImdb_v1.tar.gz
```

## Running

Both notebooks should be run from this directory so that relative paths to `aclImdb/` resolve correctly.

- **`SentimentScope_complete.ipynb`** — runs in ~13 minutes on an RTX 4080 Laptop GPU (8 epochs + 5-epoch full retrain). Run cells in order through training; the checkpoint and inference cells expect the working directory to be this folder (same as `aclImdb/` paths).
- **`SentimentScope_bert_like.ipynb`** — runs in ~30 minutes (3 epochs fine-tuning). BERT pretrained weights are downloaded automatically by HuggingFace's `transformers` library on first run.

A CUDA-capable GPU is strongly recommended. Training on CPU is possible but will be significantly slower.
