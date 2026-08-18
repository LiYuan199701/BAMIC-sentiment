# IMDB Benchmark Notebooks for the BAMIC Paper

Second benchmark dataset (after `sst2_benchmark`): the **IMDB Large Movie Review
Dataset** (Maas et al., 2011) — 50,000 full-length movie reviews, the standard
long-document binary sentiment benchmark. Long documents are BAMIC's natural setting
(like the wine corpus), so this is where the accuracy gap to the baselines is expected
to shrink.

## Run order

| # | Notebook | Model | Approx. Colab T4 time |
|---|----------|-------|------------------------|
| 00 | `00_prepare_imdb_data.ipynb` | (data prep — run first, once) | ~5–8 min |
| 01 | `01_bamic_imdb.ipynb` | BAMIC, frozen embeddings, ep10, 1 head | ~40–60 min |
| 01b | `01b_bamic_imdb_unfrozen.ipynb` | BAMIC, fine-tuned embeddings, ep5, 4 heads | ~60–90 min |
| 02 | `02_logreg_bow_imdb.ipynb` | TF-IDF logistic regression | ~2 min |
| 03 | `03_cnn_imdb.ipynb` | CNN, frozen GloVe | ~10–20 min |
| 04 | `04_bilstm_imdb.ipynb` | BiLSTM, frozen GloVe | ~20–30 min |
| 05 | `05_bert_base_imdb.ipynb` | bert-base-uncased | ~1.5–2 h |
| 06 | `06_roberta_base_imdb.ipynb` | roberta-base | ~1.5–2 h |
| 07 | `07_deberta_v3_base_imdb.ipynb` | deberta-v3-base (optional) | ~3–4 h |

Only 00 must run first; 01–07 are independent. 07 is optional — BERT + RoBERTa already
cover the transformer tier, and DeBERTa is slow at 256 tokens.

## Data

Notebook 00 downloads `aclImdb_v1.tar.gz` (84 MB) **directly from Stanford**
(`ai.stanford.edu/~amaas/data/sentiment/`) — no Hugging Face account or token.
Splits: 5% of the official 25k train split becomes validation; the official 25k test
split is the test set → train 23,750 / valid 1,250 / test 25,000, seed 20260526.
HTML line breaks (`<br />`) are stripped. If the three CSVs already exist in
`imdb_benchmark/data/`, notebook 00 skips the download entirely.

## Differences from the SST-2 setup (all deliberate)

- **MAX_LEN 256** (word tokens for static models, subwords for transformers); covers
  ~2/3 of reviews fully, the rest are truncated — this stress-tests the scalability
  limitation discussed in the paper.
- **MIN_FREQ 5** for the vocabulary (IMDB's vocabulary is much larger than SST-2's).
- **Batch sizes reduced** for the longer sequences: 64 (static models), 16
  (BERT/RoBERTa), 8 (DeBERTa). Transformers fine-tune for **2 epochs** (standard for
  IMDB's size).
- **Stopword list uses whole-word contractions** ("it's", "don't", …) because IMDB text
  is not pre-tokenized — unlike SST-2, where contractions split into fragments
  ("it 's" → "s"). The fragments are kept defensively, plus `br` for HTML residue.
- **BAMIC final uncertainty summaries use 50 posterior samples** (not 100) to keep the
  pass over 25k long reviews tractable.
- λ_S stays at the SST-2-preferred 0.02; the wine corpus (also long documents)
  preferred 0.01, so a quick {0.01, 0.02} check on IMDB is a worthwhile extra run.

## Everything else matches SST-2

Same seed (20260526), same metrics (accuracy, F1, AUC, Brier, NLL, ECE), same output
format (`outputs/<experiment>/final_metrics.csv` + history + predictions + BAMIC
uncertainty tables), same Drive layout (`MyDrive/AMIC project/imdb_benchmark/`), same
GloVe file (`word_embedding/glove.6B.300d.txt`), and the transformer notebooks carry
the same stability guards (unshuffled train evaluation loader, Adam eps 1e-6, NaN
alarm, DeBERTa at lr 5e-6). Both BAMIC notebooks include validation-tuned decision
thresholds and the full word-level posterior analysis.
