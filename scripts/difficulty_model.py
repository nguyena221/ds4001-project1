# these help us find our files, save the results, and keep track of package versions.
# the file hash lets us check later that we're using the same input file.
from pathlib import Path
import csv
import hashlib
import json
import platform
from importlib.metadata import version

# numpy handles our tables of review numbers and actual ratings.
import numpy as np
import matplotlib
# save the graphs without opening a window we have to close before the code can finish.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# logisticit is the model we're using to predict difficulty. we're just loading the tool here.
# the other tools check how far off our guesses are and show the mistakes in a graph.
from mord import LogisticIT
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# find the project folder so these paths still work if our terminal is somewhere else.
# load the cleaned version from colab, without chunking. minilm already made these numbers.
root = Path(__file__).resolve().parents[1]
input_path = root / "data/embeddings_split.npz"
output_dir = root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# open the file we downloaded from colab and pull out the three groups.
# each review is represented by 384 numbers. those numbers aren't ratings or individual words.
# the actual difficulty ratings are kept separately, so we have an answer key to compare against.
# the training group teaches the model, the validation group helps us pick settings,
# and the test group lets us check how the chosen model performs.
with np.load(input_path, allow_pickle=False) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

# check that we have 384 numbers per review and one rating to go with each review.
# the rows also need to stay in the same order from colab so we don't mix up the answers.
for features, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
    if features.ndim != 2 or features.shape[1] != 384:
        raise ValueError("Expected 384 embedding features per review.")
    if labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("Each embedding must have one matching label.")
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values.")
# check first so something wrong like 2.5 doesn't just get turned into 2 without us noticing.
for labels in (y_train, y_val, y_test):
    if not np.isin(labels, [1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be whole numbers from 1 to 5.")

# the model needs ratings written as whole numbers, so we change values like 3.0 into 3.
# we're not changing anyone's rating, just how python stores it. leave the review numbers alone.
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# print the sizes to make sure everything loaded the way we expected.
# a shape of (3289, 384) means we have 3289 reviews with 384 numbers describing each one.
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# try these alpha settings and see which gives us the lowest validation error.
# alpha controls how much training penalizes large model weights. a bigger alpha means a stronger penalty.
# it's not one of the weights, so alpha 0.5 does not mean multiply all the review numbers by 50%.
alphas = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]

# start with infinity so the first result can beat it. after that, keep looking for a lower error.
# we haven't picked a model yet, and we'll use the empty list to save each alpha's result.
best_mae = float("inf")
best_alpha = None
best_model = None
tuning_results = []

for alpha in alphas:
    # start a fresh model for each alpha and give it the training reviews plus their real ratings.
    # during training, the model figures out the 384 weights and the cutoffs between ratings.
    # we don't choose those weights ourselves.
    # each training review counts equally here. we're not giving rare ratings extra importance.
    difficulty_model = LogisticIT(alpha=alpha)
    difficulty_model.fit(X_train, y_train)

    # let the model guess the validation ratings without giving it the answers.
    # then compare with the real ratings. guessing 4 when the answer is 2 means we're off by 2.
    # mean absolute error is the average of those distances. lower is better.
    # this tells us how far off the predictions are, not the percentage we got right.
    val_predictions = difficulty_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    tuning_results.append({"alpha": alpha, "validation_mae": float(val_mae)})

    print(f"Alpha: {alpha} | Validation MAE: {val_mae:.4f}")

    # if this one has a lower validation error, save it as our best model so far.
    # if it's a tie, keep the first one. we're not using test scores to pick alpha here.
    if val_mae < best_mae:
        best_mae = val_mae
        best_alpha = alpha
        best_model = difficulty_model

print(f"\nBest alpha: {best_alpha}")
print(f"Best validation MAE: {best_mae:.4f}")

# use the best model for the graph, not just whichever alpha happened to run last.
best_val_predictions = best_model.predict(X_val)

# make a graph of the guesses versus the real answers.
# the rows show the actual ratings, and the columns show the predicted ratings.
# numbers on the diagonal are the ones we got right. the other boxes show our mistakes.
# show all five ratings, even if the model didn't guess one of them at all.
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

# save the graph in output, then close it so it doesn't hang around in memory.
# running this again replaces the old graph with the new one.
plt.savefig(output_dir / "validation_confusion_matrix.png", dpi=200)
plt.close()

# now check our chosen model on the test reviews. don't train it on the test answers.
# this checks the model we already picked; it isn't another round of choosing alpha.
# we did look at test results during earlier experiments, so we need to mention that in the report.
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

# compare against a super simple guess: always predict the middle training rating, which is 3 here.
# this helps us see whether reading the reviews actually beats just guessing the same rating every time.
baseline_rating = int(np.median(y_train))
# put the results in one place so we don't have to copy everything out of the terminal.
# accuracy tells us how often we were exactly right.
# mean absolute error tells us how far off we were on average.
# save the file hash and package versions too so we can tell what we used for this run.
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
# save the main results in a structured file and the alpha comparison in a table we can open later.
# these are just our scores and counts, not the students' written reviews.
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
with (output_dir / "alpha_validation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["alpha", "validation_mae"])
    writer.writeheader()
    writer.writerows(tuning_results)
print(f"Test MAE: {test_mae:.4f}; goal met: {test_mae <= 0.65}")
print(f"Results saved to {output_dir}")

# if we try something else later, keep these results and compare new settings using validation data.
