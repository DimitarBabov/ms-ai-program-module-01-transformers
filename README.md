# SentimentScope — Module 1 (Transformers)

IMDB binary sentiment classification with a custom **DemoGPT** transformer (from scratch) and a **BERT-identical** model with pretrained weights.

## What’s in this repo (GitHub)

| File | Description |
|------|-------------|
| `SentimentScope_complete.ipynb` | From-scratch DemoGPT (~9.5M params, ~87% test). Saves `demogpt_imdb.pt` locally after full retrain; includes a random-batch inference cell. |
| `SentimentScope_bert_like.ipynb` | BERT-style architecture + `bert-base-uncased` weights (~109M params, ~93.6% test). Saves `demogpt_bertlike.pt` locally during training. |
| `SentimentScope_Research_Report.md` | Full write-up: methods, results, and takeaways. |
| `hyperparameter_optimization_log.txt` | Optuna + manual experiment logs. |
| `requirements.txt` | Python dependencies. |

**Not tracked in git** (see `.gitignore`): `aclImdb/`, `*.pt`, `*.tar.gz`, `*.gz` — keeps the repo small and avoids Udacity’s **>1,000 files per zip** limit. Clone this repo, then add the dataset archive yourself (next section).

## Udacity / coursework zip

Include **`aclImdb_v1.tar.gz`** next to the notebooks (same folder as this README). Do **not** zip the extracted `aclImdb/` tree (tens of thousands of files). Both notebooks extract the archive automatically when `aclImdb/` is missing.

## Setup

```bash
cd ms-ai-program-module-01-transformers   # or your clone root
pip install -r requirements.txt
```

Place `aclImdb_v1.tar.gz` in this directory if it is not already there.

## Running

Run Jupyter from **this directory** so paths `aclImdb/...` resolve correctly.

1. Open `SentimentScope_complete.ipynb` → **Run All** (or top to bottom). First cells unpack the dataset if needed; training creates `demogpt_imdb.pt` locally.
2. Open `SentimentScope_bert_like.ipynb` → **Run All**. Hugging Face downloads tokenizer/BERT weights on first use (internet required). Training writes `demogpt_bertlike.pt` locally.

**Hardware:** GPU strongly recommended; CPU works but is much slower.  
**Approx. runtime (reference):** ~13 min complete notebook, ~30 min BERT-like on an RTX 4080 Laptop GPU — your machine will vary.

## Rubric checklist

- **Test accuracy >75%** — Reported in both notebooks (`SentimentScope_complete.ipynb`, `SentimentScope_bert_like.ipynb`).
- **Written report + takeaways** — `SentimentScope_Research_Report.md` (summary tables, insights sections).

## Submission tag

Stable snapshot for reviewers: git tag `module-01-final-submission` on `main` (adjust if you retag after updates).
