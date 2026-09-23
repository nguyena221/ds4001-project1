# ds4001-project1

# Data

The CourseForum CSV is excluded from GitHub.
Authorized project members should obtain it from our restricted
shared folder and place it in this directory before running the scripts.

## Selected model

The retained modeling implementation is `scripts/difficulty_model.py`: cleaned, non-chunked all-MiniLM-L6-v2 embeddings followed by `mord.LogisticIT`. Each review has 384 embedding features; training, validation, and test contain 3,289, 705, and 705 reviews. The selected alpha is 0.3.

| Metric | Validation | Test |
|---|---:|---:|
| Mean Absolute Error (MAE) | 0.7433 | 0.7461 |
| Exact-match accuracy | 39.6% | 39.0% |

The test MAE goal of 0.65 was not met. Read `output/model_results.md` for interpretation and evaluation limitations.

### Run

Keep the private cleaned, non-chunked export at `data/embeddings_split.npz`. It must contain `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, and `y_test` in matching row order. This script consumes the existing split; it does not scrape, clean, split, or generate embeddings. The upstream cleaning must remove January-term headers as well as other scraped webpage metadata. Do not substitute the old or chunked embedding export.

With the project virtual environment activated:

```powershell
python -m pip install -r requirements-model.txt
python scripts/difficulty_model.py
```

The script compares nine alpha values on validation MAE, then evaluates the selected training-set model on test data without refitting on validation. Results, alpha scores, and both confusion matrices are saved in `output/`. Re-running overwrites those generated results. Comments explain each step in beginner-friendly language.

Private CSV and NPZ datasets are excluded from new Git commits; obtain inputs through the restricted team sharing process. The existing scraper and other teammates' work are retained. No additional scraping is needed for this model.

## Preparing embeddings: VS Code or Colab

`scripts/prepare_review_embeddings.py` organizes Zilan's preprocessing and split procedure with the accepted January-header fix. It uses direct, non-chunked MiniLM encoding for all three splits. It does not train LogisticIT. Its Colab counterpart is `notebooks/prepare_review_embeddings.ipynb`, with the same cleaning, split, and embedding functions plus upload/download cells.

In Colab, choose **File → Upload notebook**, select that `.ipynb`, and run its numbered sections in order. This opens a new notebook; it does not update Zilan's original Drive file automatically. Share the organized copy with the team or transfer its sections into the agreed shared notebook. Clear notebook outputs before committing so private review contents cannot be included accidentally.

For local preparation, install `requirements-embeddings.txt` in an environment compatible with Sentence Transformers. Embedding dependencies are separate from the already-tested modeling environment; installing them can change package versions. The original Colab embedding package versions were not recorded, so regeneration may cause small numerical differences across environments.

```powershell
python -m pip install -r requirements-embeddings.txt
python scripts/prepare_review_embeddings.py --check-only
python scripts/prepare_review_embeddings.py --output data/embeddings_split_regenerated.npz
```

The check-only command validates cleaning and splitting without loading MiniLM. The generation command writes a separate private export, preserving the selected input. The default output is `data/embeddings_split.npz`, but the script refuses to overwrite an existing file unless `--overwrite` is explicitly supplied. Keep the same source CSV and row order to reproduce split membership. The current selected embedding file and model results were not regenerated during this organization work.
