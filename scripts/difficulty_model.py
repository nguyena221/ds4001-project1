# Major component: Model training and evaluation.

# Train and compare LogisticIT models using the cleaned review embeddings.
# Choose alpha using validation error, then check the selected model on the test group.

# Input: data/embeddings_cleaned_new.npz.
# Output: model scores, the alpha comparison, and confusion matrices in output/.

# These tools load files, save results, and record which data and package versions we used.
from pathlib import Path
import csv
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import matplotlib
# Save graphs without opening a separate window during the run.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mord import LogisticIT
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# Load the newly cleaned review numbers from our data folder.
root = Path(__file__).resolve().parents[1]
# Change this path if you save the embeddings with a different name or in another folder.
input_path = root / "data/embeddings_cleaned_new.npz"
output_dir = root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Each review has 384 numbers describing it, plus its actual difficulty rating.
with np.load(input_path, allow_pickle=False) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    # These are the actual ratings, kept in the same order as the review embeddings.
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

# Check that every review has 384 valid numbers and one matching rating.
for features, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
    if features.ndim != 2 or features.shape[1] != 384:
        raise ValueError("Expected 384 embedding features per review.")
    if labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("Each embedding must have one matching label.")
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values.")
# Check ratings before converting them so a value like 2.5 cannot silently become 2.
for labels in (y_train, y_val, y_test):
    if not np.isin(labels, [1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be whole numbers from 1 to 5.")

# Change ratings like 3.0 into whole numbers without changing their values.
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# Show the group sizes and a comparison table so the run is easy to follow.
print("COURSE DIFFICULTY PREDICTION | LogisticIT")
print("Cleaned review text | MiniLM embeddings | 384 features per review")
print(f"Reviews: {len(y_train):,} training | {len(y_val):,} validation | {len(y_test):,} test")
print("\nAlpha comparison (selected using validation results only)")
print(f"{'Alpha':>8} {'Validation MAE':>18}")
print("-" * 27)

# Try different alpha settings and choose using validation error.
# Alpha controls the penalty on large model weights. A larger alpha means a stronger penalty.
alphas = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]

# Start with an error of infinity so the first model can become our best result.
best_mae = float("inf")
best_alpha = None
best_model = None
tuning_results = []

for alpha in alphas:
    # Start a fresh model for each alpha and train it using the reviews and their actual ratings.
    # The model learns its own weights; we do not set those weights ourselves.
    difficulty_model = LogisticIT(alpha=alpha)
    difficulty_model.fit(X_train, y_train)

    # Compare guesses with real ratings. Mean absolute error is how far off we are on average.
    val_predictions = difficulty_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    # Keep every alpha score so we can save the full comparison, not just the winning setting.
    tuning_results.append({"alpha": alpha, "validation_mae": float(val_mae)})

    print(f"{alpha:>8.2f} {val_mae:>18.4f}")

    # Keep the model with the lowest validation error. If scores tie, keep the first one.
    if val_mae < best_mae:
        best_mae = val_mae
        best_alpha = alpha
        best_model = difficulty_model

# Use the selected model for the graph, which may not be the last model trained.
best_val_predictions = best_model.predict(X_val)

# Graph our best model. Rows are actual ratings and columns are predicted ratings.
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

# Save the validation graph and close it before creating the test graph.
plt.savefig(output_dir / "validation_confusion_matrix.png", dpi=200)
plt.close()

# Check the chosen model on the test group without training on its answers.
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

# Compare our model with always guessing the middle training rating.
baseline_rating = int(np.median(y_train))
# Save the scores and details so we know which data and settings produced these results.
metrics = {
    "model": "LogisticIT",
    "input_version": "cleaned text without chunking",
    "input_file": input_path.name,
    # This file identifier helps us check whether another run used the same embeddings.
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "selected_alpha": best_alpha,
    "validation_mae": float(best_mae),
    # Accuracy counts exact matches; MAE measures how far the predictions are from the real ratings.
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
    # Keep the earlier test inspection visible when reporting these results.
    "evaluation_caveat": "Original model test MAE 0.7645 was inspected before later validation-guided experiments; this is not a wholly untouched confirmatory test.",
    "versions": {name: version(name) for name in ["numpy", "scikit-learn", "mord", "scipy", "matplotlib"]},
    "python_version": platform.python_version(),
}
# Save exact results and the alpha comparison. Running again replaces these files.
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
with (output_dir / "alpha_validation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["alpha", "validation_mae"])
    writer.writeheader()
    writer.writerows(tuning_results)
# Print a readable summary with the goal, comparison score, and evaluation limitation.
summary = (
    "SELECTED MODEL RESULTS\n"
    f"Selected alpha: {best_alpha}\n"
    f"Validation mean absolute error: {best_mae:.4f}\n"
    f"Test mean absolute error:       {test_mae:.4f}\n"
    f"Test exact-match accuracy:      {metrics['test_accuracy']:.1%} "
    f"({metrics['correct_test_predictions']} of {len(y_test)} reviews)\n"
    f"Baseline test error (always predict {baseline_rating}): {metrics['test_baseline_mae']:.4f}\n"
    f"Target: mean absolute error of {metrics['goal_mae']:.2f} or lower | "
    f"{'Met' if metrics['goal_met'] else 'Not met'}\n\n"
    "Mean absolute error measures the average distance from the actual rating; lower is better.\n"
    "Accuracy measures the percentage of ratings predicted exactly.\n"
    "Evaluation note: test results were inspected during earlier experiments,\n"
    "so this is not a wholly untouched final evaluation.\n"
)
print("\n" + summary)
print("Results saved in output/ (metrics, alpha comparison, and confusion matrices).")
