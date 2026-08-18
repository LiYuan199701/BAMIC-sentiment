# BAMIC: Bayesian Attention-based Multiple Instance Classification

Code, experimental configurations, and evaluation procedures for the paper
*"BAMIC: Interpretable Sentiment Classification with Bayesian Uncertainty and
Word-Level Contributions"* (under review at *Expert Systems with
Applications*).

BAMIC is a compact, intrinsically interpretable sentiment classifier. Two
deterministic self-attention branches build contextual token representations
for gating and sentiment; two Bayesian scalar heads produce a soft gate score
and a signed sentiment score per token. Gate-mass normalization makes the
token contributions sum exactly to the document logit, so every prediction
comes with an exact additive word-level explanation and posterior uncertainty.

## Repository layout

```
wine_benchmark/    Wine-review corpus: data preparation, BAMIC, and baselines
                   (logistic regression BoW, CNN, BiLSTM, BERT-base, RoBERTa-base).
                   01d_bamic_wine_gatefix_bestval.ipynb is the final BAMIC model
                   reported in the paper.
sst2_benchmark/    SST-2: same protocol and baselines (+ DeBERTa-v3-base).
imdb_benchmark/    IMDB: same protocol and baselines (+ DeBERTa-v3-base).
yelp_benchmark/    Yelp Review Polarity: same protocol and baselines.
multiseed/         Ten-seed stability analysis, prior-scale sensitivity, and an
                   independent validation script (paper Appendix C.4-C.5).
*/results/         Small derived comparison tables (CSV) reported in the paper.
```

## How to run

All experiments are Jupyter notebooks designed for a Google Colab GPU runtime
(T4/L4 class). For each dataset: run `00_prepare_*_data.ipynb` first (it
downloads/organizes the public dataset and writes fixed train/validation/test
splits), then the model notebooks in numeric order. The BAMIC notebooks expect
pre-trained GloVe embeddings (`glove.6B.300d.txt`) in the project's
`word_embedding` folder, downloadable from the Stanford NLP GloVe
distribution:

```
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt -d word_embedding/
```
 Each run writes its full configuration to a
`run_config.csv` artifact (exact Python/PyTorch/CUDA versions, seeds, and
hyperparameters), so every reported number is traceable to its environment.

The multi-seed study (`multiseed/`) retrains the final Wine model for ten
preregistered seeds through the unmodified official notebook and decomposes
predictive variability into within-fit posterior and between-seed components;
`validate_claude_notebooks_v2.py` independently recomputes all aggregate
statistics from the raw per-seed outputs.

## Data

No dataset is redistributed here. SST-2, IMDB, and Yelp Review Polarity are
publicly available from the sources cited in the paper; the preparation
notebooks fetch/build them. The wine-review corpus was compiled from Wine
Spectator reviews as described in the paper's cited source; `00_prepare_wine_data.ipynb` documents
the expected input schema and the full cleaning and splitting pipeline.

## Environment

Google Colab GPU runtimes (Tesla T4 / L4), Python 3.x, PyTorch 2.x with CUDA,
scikit-learn, NumPy, pandas, matplotlib, NLTK. See `environment.md` and each
run's `run_config.csv` for exact versions.
