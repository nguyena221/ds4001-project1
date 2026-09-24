# Selected model and results

## Final retained approach

The retained model uses cleaned, non-chunked all-MiniLM-L6-v2 review embeddings and `mord.LogisticIT`. The text-cleaning correction removes leftover January-term headers and displayed average ratings before embeddings are generated. No chunk averaging or sample weighting is used. The script consumes the existing stratified 70/15/15 split: 3,289 training reviews, 705 validation reviews, and 705 test reviews, with 384 features per review.

## Results for the report

We compared alpha values of 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, and 3.0 using validation Mean Absolute Error (MAE). Alpha 0.3 achieved the lowest validation MAE, 0.7433. Without refitting on validation data, the selected model achieved a test MAE of 0.7461 and exact-match accuracy of 39.0% (275 of 705 reviews). Always predicting the training median rating of 3 had test MAE 0.9191. The model improved on that baseline but did not meet the target test MAE of 0.65 or lower.

The test confusion matrix shows difficulty identifying extreme ratings. The model correctly predicted 11 of 73 reviews rated 1 and 9 of 77 reviews rated 5. Most reviews rated 1 were predicted as 2 (40 reviews), while most reviews rated 5 were predicted as 4 (51 reviews). The model predicts all five ratings, but its predictions concentrate in the middle categories.

## Methods and evaluation limitations

- The model family and 384-dimensional embeddings follow the project overview. The alpha grid was expanded beyond the original four values; describe that expansion in the methods.
- The final selected input includes the January-header cleaning correction. The upstream notebook's other cleaning operations should be described as actually implemented; the local modeling script does not perform lowercase, HTML, or Markdown cleanup itself.
- Other models and preprocessing variants were explored during development. Their scripts and outputs were removed from the current repository to leave one selected implementation, but this does not erase that experimental history.
- Test performance was inspected during development, including for other versions. These scores are development-stage evaluation, not a wholly untouched confirmatory test. The model should not be selected or tuned further merely to lower this test score.
- MAE means Mean Absolute Error. It measures average distance from the correct rating, not percentage accuracy. A test MAE of 0.7461 means predictions were off by about three quarters of a rating step on average.
- The selected model is not fine-tuned MiniLM: MiniLM provides fixed numerical representations, and LogisticIT learns the prediction coefficients and rating boundaries.

## Reproduction and files

Run `python scripts/difficulty_model.py` using the project's virtual environment. Tested package versions are listed in `requirements-model.txt`; the recorded Python version is 3.14.7.

Private input: `data/embeddings_cleaned_new.npz` (cleaned, non-chunked export). Its expected SHA-256 is `61ab35c0d538d52dabefa8eb6956eb2677ec3c4830e1a1b1d525862e599cd0e2`. Obtain it through the restricted team sharing process. Each embedding row must match the rating in the same position. The raw review data and derived embeddings should remain private.

Outputs in this folder:

- `alpha_validation.csv`: each alpha and its full-precision validation MAE.
- `metrics.json`: metrics, accuracy, baseline, input fingerprint, package versions, and confusion-matrix counts.
- `validation_confusion_matrix.png`: validation predictions for the selected alpha.
- `test_confusion_matrix.png`: test predictions for the selected alpha.

Rows in both charts are actual ratings; columns are predicted ratings. Running the script recreates the model and overwrites these output files. It does not save a model binary or regenerate embeddings.

The installed mord/SciPy combination emits a warning about an unsupported `disp` display option. The model run completes; this warning is kept visible. The saved numerical results were verified against the earlier selected-version evaluation.

The cleaned embedding file is retained locally but removed from Git tracking. An embedding file existed in earlier Git history; the cleanup does not rewrite history.
