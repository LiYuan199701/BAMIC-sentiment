# Wine Review Benchmark for the BAMIC Paper

The **original wine corpus** — where the whole BAMIC project started — re-packaged
into the exact framework used by the SST-2 / IMDB / Yelp benchmarks, so all five
datasets produce one consistent comparison format (`final_metrics.csv` per model,
same metrics, same seed, same folder layout).

## Run order

| # | Notebook | Model | Approx. T4 time |
|---|----------|-------|------------------|
| 00 | `00_prepare_wine_data.ipynb` | data prep (run first, once) | ~3–5 min |
| 01 | `01_bamic_wine.ipynb` | BAMIC (frozen, ep10, λB=0.10, λS=0.01, warmup 2) | ~10–15 min |
| 02 | `02_logreg_bow_wine.ipynb` | TF-IDF logistic regression | ~1 min |
| 03 | `03_cnn_wine.ipynb` | Kim CNN, frozen GloVe | ~5–10 min |
| 04 | `04_bilstm_wine.ipynb` | BiLSTM, frozen GloVe | ~10–15 min |
| 05 | `05_bert_base_wine.ipynb` | fine-tuned BERT-base | ~20–25 min |
| 06 | `06_roberta_base_wine.ipynb` | fine-tuned RoBERTa-base | ~20–25 min |

(No DeBERTa, per the SST-2/IMDB experience.) 01–06 are independent; 00 first.

## Data

Notebook 00 reads `AMIC project/data/05_16_9reviewer.xlsx` (already in Drive — no
download): 141,904 Wine Spectator reviews from 9 reviewers. Label rule from the
original notebooks: **positive = rating ≥ 90**, giving a **~34% positive,
imbalanced** dataset — the one structural difference from the other benchmarks
(BAMIC/CNN/BiLSTM compensate via pos_weight ≈ 1.9; judge all models on F1/AUC
alongside accuracy). Splits: a **seeded stratified
subsample** — train 25,000 / valid 3,000 / test 10,000 (seed 20260526, verified;
positive rate 0.342 preserved in every split). The full corpus made BERT take
~40 min/epoch; the subsample mirrors the Yelp/IMDB scale, and the June full-data
BAMIC run (0.8620) remains the original-corpus reference. Notebook 00 detects and
regenerates any old full-size CSVs automatically. Reviews are short: mean ~32
words, everything ≤ 100 tokens, so MAX_LEN 100 covers the corpus completely.

Text keeps the trailing metadata ("Drink now. 175 cases made.") — detected on
~100% of reviews, ~21% of tokens — because BAMIC penalizes it explicitly and the
baselines should see the same input.

## The wine-specific piece: the boilerplate penalty (λ_B)

Notebook 01 restores the trailing-boilerplate machinery from
`Bayesian_AMIC_WD_clean.ipynb`: the regex patterns detect the metadata tail of each
review, and a second selection penalty (λ_B = 0.10, the June grid winner) pushes
Layer 1 away from selecting those tokens, with stopword positions excluded from
double-penalization. Penalty config matches the published run (which on full data reached test acc
0.8620 / AUC 0.9456 / ECE 0.0706, metadata contamination 7 → 0), plus the
now-standard 2-epoch penalty warmup, with batch 128 so the 25k subsample keeps
~195 optimizer steps per epoch.

**Watch the training log:** after warmup, `bp_delta` should fall fastest,
`sw_delta` next, and `content_delta` must stay well above both.

## Shared conventions

Same seed (20260526), same metrics (acc/F1/AUC/Brier/NLL/ECE + posterior
uncertainty for BAMIC), same Drive layout (`MyDrive/AMIC project/wine_benchmark/`),
same GloVe file, `train_eval_loader` fix in all torch baselines, eps=1e-6 + NaN
alarm in the transformer notebooks, MIN_FREQ 2 to match the original wine vocab.
