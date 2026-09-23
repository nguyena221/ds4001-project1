# ds4001-project1

# Data

The CourseForum CSV is excluded from GitHub.
Authorized project members should obtain it from our restricted
shared folder and place it in this directory before running the scripts.

## Annie's selected model

The retained modeling implementation is `scripts/difficulty_model.py`: cleaned, non-chunked all-MiniLM-L6-v2 embeddings followed by `mord.LogisticIT`. Each review has 384 embedding features; training, validation, and test contain 3,289, 705, and 705 reviews. The selected alpha is 0.3.

| Metric | Validation | Test |
|---|---:|---:|
| Mean Absolute Error (MAE) | 0.7433 | 0.7461 |
| Exact-match accuracy | 39.6% | 39.0% |

The test MAE goal of 0.65 was not met. Read `output/annie_results/Annie_results.md` for interpretation and evaluation limitations.

### Run

Keep the private cleaned, non-chunked export at `data/embeddings_split.npz`. It must contain `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, and `y_test` in matching row order. This script consumes the existing split; it does not scrape, clean, split, or generate embeddings. The upstream cleaning must remove January-term headers as well as other scraped webpage metadata. Do not substitute the old or chunked embedding export.

With the project virtual environment activated:

```powershell
python -m pip install -r requirements-model.txt
python scripts/difficulty_model.py
```

The script compares nine alpha values on validation MAE, then evaluates the selected training-set model on test data without refitting on validation. Results, alpha scores, and both confusion matrices are saved in `output/annie_results/`. Re-running overwrites those generated results. Comments explain each step in beginner-friendly language.

Private CSV and NPZ datasets are excluded from new Git commits; obtain inputs through the restricted team sharing process. The existing scraper and other teammates' work are retained. No additional scraping is needed for this model.
