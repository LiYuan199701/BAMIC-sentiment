# SST-2 Benchmark Notebooks for the BAMIC Paper

Template notebooks that run **BAMIC and six standard baselines on one shared SST-2
dataset**, so every model is evaluated on exactly the same splits with exactly the same
metrics (accuracy, F1, AUC, Brier, NLL, ECE — the same set as the wine experiments).

## Run order

| # | Notebook | Model | Approx. Colab T4 time |
|---|----------|-------|------------------------|
| 00 | `00_prepare_sst2_data.ipynb` | (data prep — run first, once) | ~2 min |
| 01 | `01_bamic_sst2.ipynb` | BAMIC (Bayesian AMIC-WD), GloVe 300d | ~20–30 min |
| 02 | `02_logreg_bow_sst2.ipynb` | TF-IDF bag-of-words logistic regression | ~1 min (CPU is fine) |
| 03 | `03_cnn_sst2.ipynb` | Kim-style CNN, frozen GloVe 300d | ~5–10 min |
| 04 | `04_bilstm_sst2.ipynb` | BiLSTM, frozen GloVe 300d | ~10–15 min |
| 05 | `05_bert_base_sst2.ipynb` | Fine-tuned bert-base-uncased | ~30–45 min |
| 06 | `06_roberta_base_sst2.ipynb` | Fine-tuned roberta-base | ~30–45 min |
| 07 | `07_deberta_v3_base_sst2.ipynb` | Fine-tuned microsoft/deberta-v3-base | ~60–90 min (batch 16) |

Notebooks 01–07 are independent of each other; only 00 must run first.

## One-time Google Drive setup

The notebooks use the same Drive project folder as the wine BAMIC work:

```
MyDrive/AMIC project/
├── word_embedding/glove.6B.300d.txt   <- must already exist (it does, from the wine runs)
└── sst2_benchmark/                    <- created automatically by the notebooks
    ├── data/                          <- written by notebook 00
    └── outputs/<experiment_name>/     <- written by each model notebook
```

If your Drive project folder has a different name, edit the single `PROJECT_DIR` line in
each notebook's path cell.

## Data splits (why they look like this)

The official GLUE SST-2 **test labels are hidden**, so, following common practice:
5% of the official train split is held out as our **validation** set (~3,368 sentences),
and the official validation split (872 sentences) is used as our labeled **test** set.
Final sizes: train 63,981 / valid 3,368 / test 872. Notebook 00 fixes these splits once
with seed 20260526 (the wine seed) and saves them as CSVs; every model reads the same
files, so results are directly comparable.

## Shared design choices

- Seed 20260526 everywhere; stratified splits; vocabulary built from training data only.
- Static-embedding models (01, 03, 04): identical tokenizer, vocabulary (min freq 2),
  MAX_LEN 60, and frozen GloVe 6B 300d matrix — so differences come from the model, not
  the preprocessing.
- Transformers (05–07): each model's own tokenizer, MAX_LEN 64 subwords, lr 2e-5,
  3 epochs, 10% warmup, grad clip 1.0 — the standard GLUE fine-tuning recipe.
- BAMIC (01): same architecture and hyperparameters as the wine notebook
  (`Bayesian_AMIC_WD_clean.ipynb`), with the boilerplate penalty OFF (SST-2 has no
  trailing metadata) and the soft stopword penalty ON at λ_S = 0.01.
- Every notebook writes `final_metrics.csv` (one row per split) in the same format, so a
  future comparison notebook can build the paper table by reading
  `sst2_benchmark/outputs/*/final_metrics.csv`.

## Running from VS Code with the Colab/Jupyter extension

Each notebook is a standard `.ipynb` and works either way:

- **In Colab directly:** upload the notebook (or open it from Drive), set
  Runtime → Change runtime type → **T4 GPU**, then Run All.
- **In VS Code with the Jupyter extension:** open the notebook and connect to a Colab
  kernel; the first cells (`nvidia-smi`, RAM check, `drive.mount`) confirm you are on the
  Colab GPU runtime, exactly like the wine notebook. The `google.colab` import in the
  Drive-mount cell only works on a Colab kernel — that's expected; these notebooks are
  designed to run on Colab compute.

## Outputs per model (in `sst2_benchmark/outputs/<experiment>/`)

- `final_metrics.csv` — accuracy, F1, AUC, Brier, NLL, ECE for train/valid/test.
- `history.csv` — per-epoch training curves (neural models).
- `test_predictions.csv` — per-sentence probabilities for error analysis.
- BAMIC only: per-document uncertainty tables, wrong-vs-correct uncertainty diagnostics,
  an example word-level posterior table + interval plot, the corpus-level word sentiment
  table with stopword-contamination count, and the model weights.
