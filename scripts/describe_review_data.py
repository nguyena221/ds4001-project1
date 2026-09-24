# summarize the original dataset and save the two plots used in the data readme.
# run this from the project environment; it does not change the data or train a model.

# input: data/thecourseforum_all_reviews (1).csv.
# output: difficulty_rating_distribution.png and hours_by_difficulty.png in output/.
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# open the original reviews using their column names.
root = Path(__file__).resolve().parents[1]
with (root / "data/thecourseforum_all_reviews (1).csv").open(encoding="utf-8-sig", newline="") as handle:
    reviews = list(csv.DictReader(handle))
# count how many reviews gave each difficulty rating.
ratings = range(1, 6)
counts = [sum(float(row["difficulty_rating"]) == rating for row in reviews) for rating in ratings]
# group the reported hours by rating, then calculate the average for each group.
hours = [[float(row["hours_per_week"]) for row in reviews
          if float(row["difficulty_rating"]) == rating and row["hours_per_week"].strip()]
         for rating in ratings]
averages = [sum(group) / len(group) for group in hours]
output = root / "output"
output.mkdir(exist_ok=True)

# keep the charts at the group level so no individual review text appears in the output.
for values, title, ylabel, filename, decimals in [
    (counts, "Number of reviews by difficulty rating", "Number of reviews", "difficulty_rating_distribution.png", 0),
    (averages, "Average reported hours per week by difficulty rating", "Average reported hours per week", "hours_by_difficulty.png", 1),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(ratings), values, color="#2878a6")
    ax.bar_label(bars, labels=[f"{v:,.{decimals}f}" for v in values], padding=4)
    ax.set(xlabel="Difficulty rating", ylabel=ylabel, title=title,
           xticks=list(ratings), ylim=(0, max(values) * 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    # save each plot to the output folder, replacing its previous version.
    fig.savefig(output / filename, dpi=200)
    plt.close(fig)
print(f"Reviews: {len(reviews)}; rating counts: {counts}")
print(f"Average hours by rating: {averages}")
print("Saved both data plots in output/.")
