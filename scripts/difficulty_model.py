from pathlib import Path
import csv
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mord import LogisticIT
from sklearn.metrics import mean_absolute_error, ConfusionMatrixDisplay

# Load the embeddings and their matching ratings.
root = Path(__file__).resolve().parents[1]
input_path = root / "data/embeddings_split (1).npz"
output_dir = root / "output/annie_results"
output_dir.mkdir(parents=True, exist_ok=True)

with np.load(input_path, allow_pickle=False) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

# Check that all labels are valid whole-number ratings.
for features, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
    if features.ndim != 2 or features.shape[1] != 384:
        raise ValueError("Expected 384 embedding features per review.")
    if labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("Each embedding must have one matching label.")
    if not np.isfinite(features).all():
        raise ValueError("Embeddings contain non-finite values.")
for labels in (y_train, y_val, y_test):
    if not np.isin(labels, [1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be whole numbers from 1 to 5.")

# Convert labels to integers before training.
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# Check the number of reviews and labels.
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Compare regularization strengths using validation MAE.
alphas = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]

best_mae = float("inf")
best_alpha = None
best_model = None
tuning_results = []

for alpha in alphas:
    difficulty_model = LogisticIT(alpha=alpha)
    difficulty_model.fit(X_train, y_train)

    val_predictions = difficulty_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    tuning_results.append({"alpha": alpha, "validation_mae": float(val_mae)})

    print(f"Alpha: {alpha} | Validation MAE: {val_mae:.4f}")

    # Keep the model with the lowest validation MAE.
    # Exact ties keep the first model.
    if val_mae < best_mae:
        best_mae = val_mae
        best_alpha = alpha
        best_model = difficulty_model

print(f"\nBest alpha: {best_alpha}")
print(f"Best validation MAE: {best_mae:.4f}")

# Generate validation predictions using the selected model.
best_val_predictions = best_model.predict(X_val)

# Plot actual ratings against predicted ratings.
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

# Create the output folder if needed, then save the chart.
Path("output").mkdir(exist_ok=True)
plt.savefig(output_dir / "validation_confusion_matrix.png", dpi=200)
plt.close()

# Freeze the validation-selected model; do not refit or tune on test results.
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

from sklearn.metrics import confusion_matrix

baseline_rating = int(np.median(y_train))
metrics = {
    "input_file": input_path.name,
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "selected_alpha": best_alpha,
    "validation_mae": float(best_mae),
    "test_mae": float(test_mae),
    "test_accuracy": float(np.mean(test_predictions == y_test)),
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
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
with (output_dir / "alpha_validation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["alpha", "validation_mae"])
    writer.writeheader()
    writer.writerows(tuning_results)
print(f"Test MAE: {test_mae:.4f}; goal met: {test_mae <= 0.65}")
print(f"Results saved to {output_dir}")

# Keep test evaluation paused while experimenting.
# test_predictions = best_model.predict(X_test)
# test_mae = mean_absolute_error(y_test, test_predictions)
# print(f"\nTest MAE: {test_mae:.4f}")
# print(f"Met the MAE goal of 0.65 or lower: {test_mae <= 0.65}")
