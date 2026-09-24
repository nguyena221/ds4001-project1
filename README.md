# Predicting theCourseForum Review Difficulty Ratings

DS 4002 Project 1 | EAZ

Our project uses student reviews from theCourseForum [1] to predict the difficulty rating that each student gave. The goal is a test mean absolute error (MAE) of 0.65 or lower on the 1-to-5 rating scale. MAE measures how many rating points our predictions are off by on average [5].

We first remove leftover webpage information from the reviews. Then we use MiniLM, an already trained model, to turn each review into 384 numbers called an embedding [2]. LogisticIT learns from those numbers and the actual ratings to predict difficulty [3]. We do not train MiniLM again or split long reviews into smaller pieces. Any text beyond MiniLM's length limit is cut off.

## Repository contents

This repository contains the code and results for predicting a student's course difficulty rating from their written review. It includes the original data collection script, the steps used to clean and prepare the reviews, the LogisticIT model, and the graphs and scores used to explain our findings.

The `scripts` folder contains the Python files and preparation notebook. The preparation code cleans the reviews, separates them into training, validation, and test groups, and uses MiniLM to turn the text into numbers. The modeling script uses those numbers to train LogisticIT, compare alpha settings, and check its predictions. A separate plotting script creates the two graphs that describe the original dataset. The original scraper is kept as a record of how we collected the reviews; it is not needed to run the analysis again.

The `data` folder contains the current embeddings and a README explaining the dataset, its columns, and how it can be used. The raw review CSV needs to be obtained separately and placed in this folder before preparing the reviews or creating the data plots. The `output` folder contains the saved model scores, alpha comparison, data plots, confusion matrices, and written results report.

To get started, Section 1 explains the software and packages needed, Section 2 shows where each file belongs, and Section 3 walks through running the project. If you want to review the findings first, open [the results summary](output/results_summary.txt) or [the detailed results report](output/model_results.md).

## Section 1: Software and platform

We used Python 3.14.7 on Windows with PowerShell and VS Code for the model results below. We originally prepared the reviews in Google Colab, and the Python scripts now let us prepare the data and run the model on our own computers.

The model uses NumPy, SciPy, scikit-learn, mord, and Matplotlib. Their tested versions are listed in `requirements-model.txt` and `output/metrics.json`. Preparing the reviews also uses pandas and sentence-transformers, listed in `requirements-embeddings.txt`. The original scraper uses Selenium, webdriver-manager, and Google Chrome. It is kept as a record of the original collection and is not part of the steps below. Use the saved dataset to repeat the analysis.

The preparation package list does not specify exact versions, and we did not record the versions originally used in Colab. Because of this, creating embeddings on another computer may give slightly different numbers. An internet connection is needed to install the packages and download MiniLM the first time.

## Section 2: Documentation map

```text
ds4001-project1/
|-- README.md                              Project overview and run instructions
|-- LICENSE.md                             MIT license for the code
|-- requirements-model.txt                 Tested modeling package versions
|-- requirements-embeddings.txt            Packages for preparing embeddings
|-- scripts/
|   |-- thecourseforum_all_reviews_scraper.py   Original data collection script
|   |-- describe_review_data.py                Recreate the two data plots
|   |-- prepare_review_embeddings.py           Clean, split, and turn reviews into numbers
|   |-- review_preprocessing.ipynb             Local preparation notebook
|   `-- difficulty_model.py                    Train, select, and evaluate LogisticIT
|-- data/                                  Data documentation and model inputs
|   |-- README.md                              Data source, columns, and quality notes
|   |-- thecourseforum_all_reviews (1).csv      Original reviews and ratings (not tracked)
|   `-- embeddings_cleaned_new.npz             Prepared model inputs and ratings
`-- output/
    |-- difficulty_rating_distribution.png Review counts by rating
    |-- hours_by_difficulty.png             Average reported hours by rating
    |-- results_summary.txt                Short summary of the results
    |-- model_results.md                   Methods, findings, and limitations
    |-- metrics.json                       Exact scores and details about the input file
    |-- alpha_validation.csv               Validation scores for each alpha
    |-- validation_confusion_matrix.png    Validation predictions versus ratings
    `-- test_confusion_matrix.png          Test predictions versus ratings
```

The `.gitignore` file excludes the raw `thecourseforum_all_reviews*.csv` files, the Python environment, and local backups. The current embeddings and data documentation are included in the repository. Request the raw CSV separately from the team. See [the data README](data/README.md#license-and-access) for data permissions and sharing details.

## Section 3: Instructions for reproducing the results

### Step 1: Get the repository and original dataset

Download or clone this repository, open its folder in VS Code, and open a PowerShell terminal in that folder. All commands below start from the project folder.

Request the original dataset directly from the team. Create a folder named `data` if it does not exist, and put the file there with this exact name:

```text
data/thecourseforum_all_reviews (1).csv
```

We will share the raw CSV directly and privately with our professor. It is not included in the repository. After receiving it, place it in `data/` using the filename above. Read [the data README](data/README.md) for the data permissions and ethical statements.

Use the same dataset and row order. Collecting new reviews, sorting rows, or dropping reviews can change the split and results.

### Step 2: Set up Python and install packages

If the project does not already have a `.venv` folder, create a Python environment to hold the project's packages:

```powershell
python -m venv .venv
```

Install the modeling and preparation packages into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt -r requirements-embeddings.txt
```

These commands use the project's Python environment directly, so you do not need to activate it first. If the installation fails, fix that error before moving on.

### Step 3: Check the data and split

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_embeddings.py --check-only
```

This checks that every review has text and a valid difficulty rating. It also removes the webpage heading with the term, year, and average rating, along with the vote count and date at the end. It then divides the reviews into the groups below. This check does not create embeddings or replace any files.

| Group | Reviews | Purpose |
|---|---:|---|
| Training | 3,289 | Learn the relationship between review numbers and ratings |
| Validation | 705 | Choose the alpha setting |
| Test | 705 | Evaluate the selected model |

We put 70 percent of the reviews in training, 15 percent in validation, and 15 percent in testing. Each group has a similar mix of difficulty ratings. Using `random_state=42` repeats the same split as long as the dataset and row order stay the same [4].

### Step 4: Generate the cleaned embeddings

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_embeddings.py
```

This loads `all-MiniLM-L6-v2`, converts the cleaned reviews into 384 numbers each, and saves `data/embeddings_cleaned_new.npz`. The file contains the training, validation, and test embeddings together with their actual ratings in matching order.

If this file already exists, the script stops so it does not replace it by accident. You can keep using the checked file. To create it again and replace the old version, run:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_review_embeddings.py --overwrite
```

You only need to run the Python preparation script for these steps. The notebook now supports the same local workflow in VS Code, using files in the data folder.

If you save embeddings under a different filename or in a different folder, update `input_path` near the top of `scripts/difficulty_model.py` to match. For example, `data/embeddings_updated.npz` needs:

```python
input_path = root / "data/embeddings_updated.npz"
```

If you keep the same filename and folder, you do not need to change the code. After replacing the file or switching to a different one, run the difficulty model again to update the scores and graphs. Update the written results in the READMEs and `output/model_results.md` if the scores change.

### Step 5: Train and evaluate the difficulty model

```powershell
.\.venv\Scripts\python.exe scripts/difficulty_model.py
```

The script loads `data/embeddings_cleaned_new.npz`, trains LogisticIT on the training group, and compares alpha values of 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, and 3.0. Alpha controls how strongly the model is penalized for using large weights during training. A larger alpha means a stronger penalty.

The script keeps the model with the lowest validation MAE and checks its predictions on the test group. The validation and test ratings are used to check predictions, not to train the model. Each run trains the model again and updates the scores and graphs in `output/`. It does not save the trained model itself. The written report, `output/model_results.md`, needs to be updated separately if the results change.

You may see `Unknown solver options: disp` from the model packages. We saw this warning in our successful run, and the script still finished. If you see a different error, check it before continuing.

### Step 6: Review the results

To recreate the two dataset plots described in [the data README](data/README.md), run:

```powershell
.\.venv\Scripts\python.exe scripts/describe_review_data.py
```


Open [the results summary](output/results_summary.txt) to check the scores from your run. The expected results for the current embeddings are:

| Metric | Validation | Test |
|---|---:|---:|
| Mean absolute error | 0.7433 | 0.7461 |
| Exact-match accuracy | 39.6% | 39.0% |

The expected selected alpha is **0.3**. See [the model results report](output/model_results.md) for the detailed findings, comparison with a simple baseline, and whether the project goal was met.

The confusion matrices show where the predictions were right or wrong. Rows show the actual ratings, and columns show the predicted ratings. `output/alpha_validation.csv` lists the score for each alpha we tried. `output/metrics.json` stores the exact scores, package versions, and a file identifier called a SHA-256 hash, which helps us check whether we used the same input file.

Read [the methods and evaluation limitations](output/model_results.md#methods-and-evaluation-limitations) before interpreting or presenting these scores.

## References

[1] theCourseForum, "theCourseForum." Accessed: Sep. 23, 2026. [Online]. Available: https://thecourseforum.com/

[2] Sentence Transformers, "sentence-transformers/all-MiniLM-L6-v2," Hugging Face. Accessed: Sep. 23, 2026. [Online]. Available: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[3] F. Pedregosa, "mord: Ordinal Regression in Python," mord documentation. Accessed: Sep. 23, 2026. [Online]. Available: https://pythonhosted.org/mord/

[4] Scikit-learn developers, "train_test_split," scikit-learn documentation. Accessed: Sep. 23, 2026. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

[5] Scikit-learn developers, "mean_absolute_error," scikit-learn documentation. Accessed: Sep. 23, 2026. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
