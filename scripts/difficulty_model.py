# these built-in tools find folders, save results, and record which software versions we used.
# a file hash acts like a fingerprint so we can identify the exact input file for this run.
from pathlib import Path
import csv
import hashlib
import json
import platform
from importlib.metadata import version

# numpy works with arrays: tables of numbers representing reviews and their ratings.
import numpy as np
import matplotlib
# this makes charts save directly to files without opening a window or pausing the script.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# logisticit is the prediction model; importing it does not train it yet.
# the metric measures prediction error, and the display tool draws confusion matrices.
from mord import LogisticIT
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# locate the project folder from this script's location, rather than the terminal's location.
# the input is cleaned, non-chunked review embeddings exported from colab; minilm does not run here.
root = Path(__file__).resolve().parents[1]
input_path = root / "data/embeddings_split.npz"
output_dir = root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# open the saved bundle of arrays; the with block closes the file after loading them.
# each x row describes one review using 384 numbers, not 384 words or difficulty scores.
# the matching y entry is the difficulty rating the student actually submitted, from 1 to 5.
# training examples teach the model; validation examples choose alpha; test examples evaluate it.
with np.load(input_path, allow_pickle=False) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

# stop early if the inputs have the wrong shape, missing numerical values, or mismatched sizes.
# matching row counts are necessary, but the colab export must also preserve the correct row order.
for features, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
    if features.ndim != 2 or features.shape[1] != 384:
        raise ValueError("Expected 384 embedding features per review.")
    if labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("Each embedding must have one matching label.")
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values.")
# check rating values before conversion so an invalid value like 2.5 is not silently changed to 2.
for labels in (y_train, y_val, y_test):
    if not np.isin(labels, [1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be whole numbers from 1 to 5.")

# convert ratings such as 3.0 into integers such as 3 because mord expects integer categories.
# this changes their storage type, not their meaning; the embedding values remain decimals.
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# print shapes so we can confirm the number of reviews, embedding columns, and matching labels.
# for example, (3289, 384) means 3289 reviews, each described by 384 numbers.
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# try several alpha settings. alpha controls the penalty for large learned model coefficients.
# alpha is not one of the 384 coefficients and does not multiply every input by this amount.
# larger alpha discourages large coefficients more strongly; validation results decide what works.
alphas = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]

# start the best error at infinity so the first trained model becomes our initial best.
# none means we have not selected an alpha or trained model yet; the list records every trial.
best_mae = float("inf")
best_alpha = None
best_model = None
tuning_results = []

for alpha in alphas:
    # create a fresh model for this alpha, then learn from training reviews and their actual ratings.
    # fit automatically learns 384 coefficients plus boundaries for choosing ratings 1 through 5.
    # every training review counts equally here; no sample weighting is applied.
    difficulty_model = LogisticIT(alpha=alpha)
    difficulty_model.fit(X_train, y_train)

    # predict receives only review embeddings, not the validation answers.
    # afterward, mae compares its guesses with the actual ratings: predicting 4 instead of 2 adds 2.
    # mae averages those absolute errors across reviews; smaller is better, and it is not a percentage.
    val_predictions = difficulty_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    tuning_results.append({"alpha": alpha, "validation_mae": float(val_mae)})

    print(f"Alpha: {alpha} | Validation MAE: {val_mae:.4f}")

    # keep the entire trained model whenever its validation error is lower than our previous best.
    # exact ties keep the first model; test results play no part in this selection.
    if val_mae < best_mae:
        best_mae = val_mae
        best_alpha = alpha
        best_model = difficulty_model

print(f"\nBest alpha: {best_alpha}")
print(f"Best validation MAE: {best_mae:.4f}")

# predict again with the selected model: the loop's last model is not necessarily the best one.
best_val_predictions = best_model.predict(X_val)

# draw a confusion matrix: rows show actual ratings, columns show predicted ratings.
# diagonal cells count exact matches; other cells show the kinds of mistakes the model makes.
# explicit labels keep all five ratings visible even if the model never predicts one of them.
ConfusionMatrixDisplay.from_predictions(
    y_val,
    best_val_predictions,
    labels=[1, 2, 3, 4, 5],
    display_labels=[1, 2, 3, 4, 5],
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title(f"Validation Confusion Matrix — Alpha {best_alpha}")
plt.tight_layout()

# save the chart as an image and close it to free plotting resources.
# existing results with this filename are overwritten when this script runs again.
plt.savefig(output_dir / "validation_confusion_matrix.png", dpi=200)
plt.close()

# evaluate the already-selected model on test reviews without fitting it again.
# the test scores below evaluate this chosen version, rather than choosing another alpha.
# an earlier test result was viewed during development, so the report must disclose that limitation.
test_predictions = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
ConfusionMatrixDisplay.from_predictions(
    y_test, test_predictions, labels=[1, 2, 3, 4, 5],
    cmap="Blues", values_format="d", colorbar=False
)
plt.title(f"Test Confusion Matrix - Alpha {best_alpha}")
plt.tight_layout()
plt.savefig(output_dir / "test_confusion_matrix.png", dpi=200)
plt.close()

# a baseline is a simple prediction rule to check whether our model adds useful information.
# here it always predicts the middle training rating (the median), regardless of the review's text.
baseline_rating = int(np.median(y_train))
# collect results in a dictionary so they can be saved together in a readable json file.
# accuracy counts exact matches; mae also accounts for how far wrong predictions are from the answer.
# record the input fingerprint, software versions, and evaluation caveat for reproducibility.
metrics = {
    "model": "LogisticIT",
    "input_version": "cleaned text without chunking",
    "input_file": input_path.name,
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "selected_alpha": best_alpha,
    "validation_mae": float(best_mae),
    "validation_accuracy": float(np.mean(best_val_predictions == y_val)),
    "test_mae": float(test_mae),
    "test_accuracy": float(np.mean(test_predictions == y_test)),
    "correct_test_predictions": int(np.sum(test_predictions == y_test)),
    "goal_mae": 0.65,
    "goal_met": bool(test_mae <= 0.65),
    "baseline_rating": baseline_rating,
    "validation_baseline_mae": float(mean_absolute_error(y_val, np.full(len(y_val), baseline_rating))),
    "test_baseline_mae": float(mean_absolute_error(y_test, np.full(len(y_test), baseline_rating))),
    "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
    "validation_confusion_matrix": confusion_matrix(y_val, best_val_predictions, labels=[1, 2, 3, 4, 5]).tolist(),
    "test_confusion_matrix": confusion_matrix(y_test, test_predictions, labels=[1, 2, 3, 4, 5]).tolist(),
    "matrix_convention": "Rows actual, columns predicted; ratings 1 through 5",
    "refit_on_validation": False,
    "evaluation_caveat": "Original model test MAE 0.7645 was inspected before later validation-guided experiments; this is not a wholly untouched confirmatory test.",
    "versions": {name: version(name) for name in ["numpy", "scikit-learn", "mord", "scipy", "matplotlib"]},
    "python_version": platform.python_version(),
}
# save overall results as json and each alpha's validation score as a table that can be opened later.
# these files contain aggregate results, not the written student reviews.
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
with (output_dir / "alpha_validation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["alpha", "validation_mae"])
    writer.writeheader()
    writer.writerows(tuning_results)
print(f"Test MAE: {test_mae:.4f}; goal met: {test_mae <= 0.65}")
print(f"Results saved to {output_dir}")

# for future tuning experiments, use validation data and preserve this run as the reference version.
