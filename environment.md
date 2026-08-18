# Environment

All training and evaluation ran on Google Colab GPU runtimes (Tesla T4 and
L4 GPUs were used across runs). The analysis notebooks (multi-seed
aggregation, validation) run on CPU.

Core dependencies (all preinstalled on Colab):

- Python 3.x
- PyTorch 2.x with CUDA
- scikit-learn
- NumPy, pandas, matplotlib
- NLTK (stopword lists)

Exact versions for every training run are recorded at run time in that run's
`run_config.csv` output artifact (fields: `python_version`, `torch_version`,
`cuda_version`, `device_name`), together with the full hyperparameter
configuration and the preregistered seed plan. The independent validator
(`multiseed/validate_claude_notebooks_v2.py`) additionally requires only
NumPy, pandas, and scikit-learn.

Embeddings: GloVe 6B, 300-dimensional (`glove.6B.300d.txt`), from the
Stanford NLP GloVe distribution.
