# Selected model and results

## Method we used

We used `all-MiniLM-L6-v2` to turn the cleaned reviews into numbers and `mord.LogisticIT` to predict their difficulty ratings. We fixed the cleaning step to include January headings and remove displayed average ratings. We did not split reviews into smaller pieces or give some reviews more weight during training. Each review has 384 numbers describing it. We used 3,289 reviews for training, 705 for validation, and 705 for testing, keeping a similar mix of ratings in each group.

## Results for the report

Mean absolute error (MAE) tells us how many rating points our predictions are off by on average. Lower is better. Accuracy is the percentage of ratings we predicted exactly.

We compared alpha values of 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, and 3.0 using validation Mean Absolute Error (MAE). Alpha 0.3 achieved the lowest validation MAE, 0.7433. We then checked the same model on the test group without training it on the validation ratings. It had a test MAE of 0.7461 and exact-match accuracy of 39.0% (275 of 705 reviews). Always predicting the training median rating of 3 had test MAE 0.9191. The model improved on that baseline but did not meet the target test MAE of 0.65 or lower.

The test confusion matrix shows that the model had more trouble with ratings 1 and 5. The model correctly predicted 11 of 73 reviews rated 1 and 9 of 77 reviews rated 5. Most reviews rated 1 were predicted as 2 (40 reviews), while most reviews rated 5 were predicted as 4 (51 reviews). The model predicted all five ratings, but most of its predictions were 2, 3, or 4.

## Methods and evaluation limitations

- We used the model and 384-number embeddings described in the project overview. We tried nine alpha settings instead of the original four.
- The cleaning step removes webpage headings, including January terms, and the vote count and date at the end. We did not add separate steps to convert text to lowercase or remove HTML and Markdown, even though those were in the original plan.
- We tried other models and ways of preparing the reviews before choosing this version. We removed those extra scripts and results to keep the repository organized, but they were still part of our experiments.
- We looked at test scores during earlier experiments, so the test group was not completely unseen throughout the project. We need to include that limitation when discussing these results. Any future model comparisons should use validation results rather than choosing whichever model gives a better test score.
- MAE means Mean Absolute Error. It measures average distance from the correct rating, not percentage accuracy. A test MAE of 0.7461 means predictions were off by about three quarters of a rating step on average.
- We did not train MiniLM again. It turns reviews into numbers, and LogisticIT learns how to use those numbers to predict ratings.

## Reproduction and files

Run `python scripts/difficulty_model.py` using the project's virtual environment. Tested package versions are listed in `requirements-model.txt`; the recorded Python version is 3.14.7.

Private input: `data/embeddings_cleaned_new.npz` (cleaned, non-chunked export). Its expected SHA-256 is `61ab35c0d538d52dabefa8eb6956eb2677ec3c4830e1a1b1d525862e599cd0e2`. Use the current embeddings shared through the private team repository. Each row of review numbers must stay matched with its actual rating. The raw review data and derived embeddings should remain private.

### What each output file contains

| File | What it shows or saves |
|---|---|
| `alpha_validation.csv` | The validation MAE for each alpha we tried. This shows how we chose the setting with the lowest validation error. The saved scores are not rounded. |
| `difficulty_rating_distribution.png` | The number of reviews at each difficulty rating from 1 to 5. This shows which ratings are more common in the original dataset. |
| `hours_by_difficulty.png` | The average reported hours per week for each difficulty rating. This helps describe the dataset; hours are not used as an input to our prediction model. |
| `metrics.json` | The exact model scores, selected alpha, group sizes, comparison with always guessing the middle training rating, and confusion-matrix counts. It also records package versions and the input file identifier so we can check what was used for the run. |
| `model_results.md` | This report. It explains our method, findings, limitations, and the files in this folder. Update it manually if the results change. |
| `test_confusion_matrix.png` | The selected model's predictions compared with the actual test ratings. It shows which ratings were predicted correctly and where the model made mistakes. |
| `validation_confusion_matrix.png` | The selected model's predictions compared with the actual validation ratings. It shows the mistakes made on the group used to compare alpha settings. |

In both confusion matrices, rows show actual ratings and columns show predicted ratings. Numbers on the diagonal count the exact matches.

Running `scripts/difficulty_model.py` updates `alpha_validation.csv`, `metrics.json`, and both confusion matrices. Running `scripts/describe_review_data.py` updates the difficulty-rating and weekly-hours plots. Neither script updates this written report. The modeling script does not save the trained model itself or create the embeddings again.

The model packages print a warning about the `disp` display option. The run still finishes, and we confirmed that the scores match the earlier run of this model.

The current embeddings can be included in the private repository. The raw CourseForum CSV stays excluded. An older embeddings file also remains in Git history.
