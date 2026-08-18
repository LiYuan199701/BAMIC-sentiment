# Yelp Review Polarity Benchmark for the BAMIC Paper

Fourth benchmark, chosen as BAMIC-friendly terrain after the ERASER movies result:
our cross-benchmark diagnosis is that BAMIC needs **many training documents of
moderate length with strongly lexical sentiment**. Yelp Review Polarity (Zhang,
Zhao & LeCun, NeurIPS 2015) matches that profile even better than IMDB — 25k
medium reviews (mean ~130 words) whose sentiment is carried by individual words
("delicious", "rude", "overpriced") — while remaining a standard benchmark that
reviewers recognize.

## Run order

| # | Notebook | Model | Approx. T4 time |
|---|----------|-------|------------------|
| 00 | `00_prepare_yelp_data.ipynb` | data prep (run first, once) | ~5 min |
| 01 | `01_bamic_yelp.ipynb` | BAMIC (frozen, ep10, λS=0.02, warmup 2) | ~30–45 min |
| 02 | `02_logreg_bow_yelp.ipynb` | TF-IDF logistic regression | ~2 min |
| 03 | `03_cnn_yelp.ipynb` | Kim CNN, frozen GloVe | ~10–20 min |
| 04 | `04_bilstm_yelp.ipynb` | BiLSTM, frozen GloVe | ~20–30 min |
| 05 | `05_bert_base_yelp.ipynb` | fine-tuned BERT-base | ~1.5–2 h |
| 06 | `06_roberta_base_yelp.ipynb` | fine-tuned RoBERTa-base | ~1.5–2 h |

(No DeBERTa, per the SST-2/IMDB experience.) 01–06 are independent; 00 first.

## Data

Notebook 00 downloads `yelp_review_polarity_csv.tgz` (166 MB) directly from the
fast.ai AWS mirror — no account needed, URL verified live. From the official
560k/38k splits it takes a **seeded, class-balanced subsample**: train 25,000 /
valid 3,000 / test 10,000 (seed 20260526, fully reproducible; sizes mirror the
IMDB benchmark so runtimes and behavior are directly comparable). Labels are
mapped from Zhang et al.'s 1=neg/2=pos to y=0/1; literal `\n` escapes are cleaned.
Saved to Drive as `yelp_train/valid/test.csv` with columns `text`, `y`.

Measured on the real data: mean 134 words, median 99; MAX_LEN 256 fully covers
~88% of reviews (vs ~two-thirds truncated on IMDB at the same window).

## BAMIC configuration

Exactly the recipe that produced the IMDB calibration headline (0.8584 acc /
ECE 0.0098): frozen GloVe, 10 epochs, 1 head, λS = 0.02 with
STOPWORD_WARMUP_EPOCHS = 2, MAX_LEN 256, batch 64, lr 3e-4. All knobs live in
one config cell; the output folder is self-naming. **Watch the training log:**
`content_delta` must stay well above `stopword_delta` after the penalty switches
on at epoch 3 — if both decay toward 0 together, that is the gate-collapse
signature. With 25k documents and 391 steps/epoch the likelihood signal matches
IMDB's, where this recipe trained cleanly.

## Shared conventions

Same seed (20260526), same metrics (acc/F1/AUC/Brier/NLL/ECE + posterior
uncertainty for BAMIC), same Drive layout
(`MyDrive/AMIC project/yelp_benchmark/`), same GloVe file, same
`final_metrics.csv` format, `train_eval_loader` fix in all torch baselines,
eps=1e-6 + NaN alarm in the transformer notebooks.
