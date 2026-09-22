# Annie Nguyen: Model Architecture and Tuning

## Report-ready results

We trained an ordinal logistic regression model (`mord.LogisticIT`) using 384-dimensional all-MiniLM-L6-v2 review embeddings. The stratified training, validation, and test sets contained 3,289, 705, and 705 reviews, respectively. We compared alpha values of 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, and 3.0 using validation Mean Absolute Error (MAE). Alpha 0.5 achieved the lowest validation MAE, 0.7418. Without refitting on validation data, the selected model achieved a test MAE of 0.7433 and exact-match accuracy of 39.7% (280 of 705 reviews). A constant prediction of 3, the training-set median, had test MAE 0.9191. The model therefore improved on this baseline but did not meet the target MAE of 0.65 or lower.

The test confusion matrix showed predictions concentrated in ratings 2–4. Only 10 of 73 reviews rated 1 and 8 of 77 reviews rated 5 were predicted correctly. Of the reviews rated 1, 42 were predicted as 2; of those rated 5, 50 were predicted as 4. The model learned useful relationships between review language and difficulty but tended to move extreme ratings toward the middle of the scale.

## Evaluation limitation — include with the results

The original model's test MAE (0.7645) was inspected before subsequent validation-guided experiments. Later choices used validation MAE, but this test set is not a wholly untouched confirmatory evaluation. These results should be described as development-stage evaluation. A stronger future assessment would freeze the full pipeline and evaluate on genuinely unused data obtained through the project's permitted process. Repeatedly tuning against these test results or reshuffling already inspected data would not restore an independent final test.

## Changes from the original plan

- The original alpha grid was 0.01, 0.1, 1.0, and 10.0. A finer validation grid was subsequently used; this should be reflected in the methods section.
- The cleaning expression was extended to remove January-term headers and scraped average ratings. Removing scraped metadata is necessary for the text-only research question regardless of its effect on MAE.
- Token counting found 554 training reviews (16.8%) exceeded the embedding model's 256-token limit. The Colab experiment recursively split long reviews into pieces that fit the limit, embedded each piece, and took an unweighted mean of each review's piece embeddings. Short reviews remained intact. The existing split membership and label order were retained.
- With corrected cleaning but no chunking, best validation MAE was 0.7433 at alpha 0.3. With chunking, it was 0.7418 at alpha 0.5. This difference represents only one total absolute-error point across 705 validation reviews, so it is not strong evidence that chunking meaningfully improves generalization.
- MAE means **Mean Absolute Error**, not Mean Average Error.

## Validation comparison for the current chunked input

| Alpha | Validation MAE |
|---|---:|
| 0.03 | 0.7801 |
| 0.05 | 0.7872 |
| 0.1 | 0.7773 |
| 0.2 | 0.7674 |
| 0.3 | 0.7645 |
| 0.5 | 0.7418 |
| 1.0 | 0.7546 |
| 2.0 | 0.7574 |
| 3.0 | 0.7631 |

## Files and reproduction

- `scripts/difficulty_model.py`: training, validation selection, evaluation, and saving outputs. Paths are resolved relative to the project, so the launch directory does not matter.
- Private input: `data/embeddings_split (1).npz`, the user-exported chunked embeddings. Its SHA-256 hash and the installed package versions are recorded in `metrics.json` to identify the exact run.
- `alpha_validation.csv`: full-precision validation scores.
- `metrics.json`: selected settings, metrics, baseline results, both confusion matrices, input hash, software versions, and evaluation caveat.
- `validation_confusion_matrix.png` and `test_confusion_matrix.png`: rows are actual ratings; columns are predictions.

Run from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/difficulty_model.py
```

This command repeats the fixed evaluation and overwrites the result files. Preserve this results folder before starting any later experiment. No trained model binary is saved; rerunning reconstructs the model from the recorded input and settings. Keep a copy of the final chunking notebook with the team, since the earlier notebook supplied for inspection did not yet contain the chunking functions.

The installed `mord`/SciPy combination emits `OptimizeWarning: Unknown solver options: disp`. The run completed and reproduced the reported validation scores. This warning concerns an unrecognized display option; it is retained rather than silently suppressed.

## Team handoff draft

Annie's current modeling implementation and results are ready for review. The current chunked-embedding model selects alpha 0.5, with validation MAE 0.7418 and test MAE 0.7433. The 0.65 goal was not met. Both confusion matrices and a median-rating baseline are saved. Please update the shared methods to document the finer alpha grid, January-header cleaning, and chunk averaging, and include the limitation that the original test result was viewed before further development. Chunking provided only a tiny validation improvement. This is the current completed version, not a claim that the team has approved the method changes.

## Data handling

No data or messages were uploaded or sent during this completion. NPZ files are now ignored for future Git additions. However, `data/embeddings_split.npz` was already tracked by Git when this work began; adding an ignore rule does not remove it from tracking or history. Keep the repository private as agreed and resolve any public-release data handling with the team. No Git history or staging changes were made here.
