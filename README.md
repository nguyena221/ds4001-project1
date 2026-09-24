# ds4001-project1

# Data

The CourseForum CSV is excluded from GitHub.
Authorized project members should obtain it from our restricted
shared folder and place it in the `data/` folder before running the scripts.

## Selected model

The retained modeling implementation is `scripts/difficulty_model.py`: cleaned, non-chunked all-MiniLM-L6-v2 embeddings followed by `mord.LogisticIT`. Each review has 384 embedding features; training, validation, and test contain 3,289, 705, and 705 reviews. The selected alpha is 0.3.

| Metric | Validation | Test |
|---|---:|---:|
| Mean Absolute Error (MAE) | 0.7433 | 0.7461 |
| Exact-match accuracy | 39.6% | 39.0% |

The test MAE goal of 0.65 was not met. Read `output/model_results.md` for interpretation and evaluation limitations.

### Run

Keep the private cleaned, non-chunked export at `data/embeddings_cleaned_new.npz`. It must contain `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, and `y_test` in matching row order. This script consumes the existing split; it does not scrape, clean, split, or generate embeddings. The upstream cleaning must remove January-term headers as well as other scraped webpage metadata. Do not substitute the old or chunked embedding export.

With the project virtual environment activated:

```powershell
python -m pip install -r requirements-model.txt
python scripts/difficulty_model.py
```

The script compares nine alpha values on validation MAE, then evaluates the selected training-set model on test data without refitting on validation. Results, alpha scores, and both confusion matrices are saved in `output/`. Re-running overwrites those generated results. Comments explain each step in beginner-friendly language.

Private CSV and NPZ datasets are excluded from new Git commits; obtain inputs through the restricted team sharing process. The existing scraper and other teammates' work are retained. No additional scraping is needed for this model.

## Preparing embeddings: VS Code or Colab

`scripts/prepare_review_embeddings.py` organizes Zilan's preprocessing and split procedure with the accepted January-header fix. It uses direct, non-chunked MiniLM encoding for all three splits. It does not train LogisticIT. Zilan's Colab notebook is retained at `scripts/review_preprocessing.ipynb` for the team's notebook workflow. Use the Python script for the local workflow.

In Colab, choose **File → Upload notebook**, select that `.ipynb`, and run its sections in order. This opens a new notebook; it does not update Zilan's original Drive file automatically. Share the organized copy with the team or transfer its sections into the agreed shared notebook. Clear notebook outputs before committing so private review contents cannot be included accidentally.

For local preparation, install `requirements-embeddings.txt` in an environment compatible with Sentence Transformers. Embedding dependencies are separate from the already-tested modeling environment; installing them can change package versions. The original Colab embedding package versions were not recorded, so regeneration may cause small numerical differences across environments.

```powershell
python -m pip install -r requirements-embeddings.txt
python scripts/prepare_review_embeddings.py --check-only
python scripts/prepare_review_embeddings.py --output data/embeddings_cleaned_new.npz
```

The check-only command validates cleaning and splitting without loading MiniLM. The generation command and default output both use `data/embeddings_cleaned_new.npz`, the file loaded by the modeling script. If that file already exists, generation stops. Add `--overwrite` only when intentionally regenerating it, then rerun the modeling script to update the results. Keep the same source CSV and row order to reproduce split membership. The current selected file, `data/embeddings_cleaned_new.npz`, was regenerated locally with this preparation script and evaluated with the modeling script. Validation MAE remained 0.7433 and test MAE remained 0.7461.

## Results files

- `output/results_summary.txt`: short summary to show the professor.
- `output/model_results.md`: methods, interpretation, and limitations for the report.
- `output/metrics.json`: exact scores and input details for reproducibility.
- `output/alpha_validation.csv`: validation scores for each alpha tried.
- `output/validation_confusion_matrix.png` and `output/test_confusion_matrix.png`: where predictions were right or wrong.
