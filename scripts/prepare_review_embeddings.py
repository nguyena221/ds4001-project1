# clean the review text, split the reviews into three groups, and create MiniLM embeddings.
# these embeddings are the numbers our difficulty model uses to learn from the reviews.

# input: data/thecourseforum_all_reviews (1).csv.
# output: data/embeddings_cleaned_new.npz, with review embeddings and matching ratings.

# these tools handle local paths, command-line options, and removal of scraped text.
import argparse
from pathlib import Path
import re

# pandas reads the csv; numpy saves arrays; scikit-learn separates the review groups.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_review(text):
    """remove the same webpage headers and footers as the selected notebook version."""
    # the January option is the accepted correction; keep the rest of the rule unchanged.
    # this removes a term, year, and displayed average rating before the student's writing.
    text = re.sub(
        r'^(Fall|Spring|Summer|Winter|January)\s+\d{4}\s+\d+(?:\.\d+)?\s+AVERAGE\s*',
        '', text, flags=re.IGNORECASE
    )
    # this removes a trailing vote count and date, not sentences within the review.
    text = re.sub(
        r'\s+-?\d+\s+'
        r'(?:Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|'
        r'Sept?\.?|Oct\.?|Nov\.?|Dec\.?|January|February|March|April|'
        r'June|July|August|September|October|November|December)'
        r'\s+\d{1,2},\s+\d{4}\s*$',
        '', text, flags=re.IGNORECASE
    )
    return text.strip()


def prepare_splits(csv_path):
    """load, clean, and split the reviews without changing their original order."""
    reviews = pd.read_csv(csv_path)
    required = {"review_text", "difficulty_rating"}
    if not required.issubset(reviews.columns):
        raise ValueError("The CSV must contain review_text and difficulty_rating.")
    # fail on unexpected data instead of silently dropping rows and changing the saved split.
    # the selected dataset has no missing review texts or difficulty ratings.
    if reviews[list(required)].isna().any().any():
        raise ValueError("Missing reviews or ratings: resolve these before reproducing the split.")
    if not reviews["review_text"].map(lambda text: isinstance(text, str)).all():
        raise ValueError("Every review must be text.")
    if not reviews["difficulty_rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Difficulty ratings must be whole numbers from 1 to 5.")

    # apply one definitive cleaning function, rather than cleaning twice with different rules.
    # no extra lowercase, punctuation, or markdown transformations are added to this selected version.
    reviews["cleaned_review"] = reviews["review_text"].apply(clean_review)
    if reviews["cleaned_review"].eq("").any():
        raise ValueError("Cleaning produced an empty review; inspect the input first.")
    X = reviews["cleaned_review"]
    y = reviews["difficulty_rating"]

    # reserve 30% temporarily, leaving 70% for training.
    # stratify preserves approximate rating proportions; 42 makes the selection repeatable.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # divide the remaining 30% into equal validation and test groups (15% each overall).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for name, (texts, labels) in splits.items():
        if not texts.index.equals(labels.index):
            raise ValueError(f"Review and label indexes do not match for {name}.")
        print(f"{name}: {len(texts)} reviews")
        print(labels.value_counts().sort_index().to_string())
    return splits


def save_embeddings(splits, output_path):
    """use the pretrained model to encode whole reviews, then save the six model inputs."""
    # import the large language-model package only when we actually need embeddings.
    # the check-only option can therefore inspect cleaning and splits without downloading MiniLM.
    from sentence_transformers import SentenceTransformer

    # this uses MiniLM's existing knowledge; it does not train or fine-tune MiniLM.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    arrays = {}
    for name, (texts, labels) in splits.items():
        # use the same direct encode call for all three groups; there is no chunking here.
        # long reviews are truncated according to the model's default input limit.
        features = model.encode(texts.tolist(), show_progress_bar=True)
        if features.shape != (len(labels), 384) or not np.isfinite(features).all():
            raise ValueError(f"Unexpected embedding shape or values for {name}.")
        arrays[f"X_{name}"] = features
        arrays[f"y_{name}"] = labels.to_numpy()
        print(f"{name} embeddings: {features.shape}; labels: {labels.shape}")
    # keep row order intact: the nth embedding belongs to the nth rating in the same split.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"Saved private embeddings to {output_path}")


def main():
    # paths default to this project's data folder, even when launched from another directory.
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/thecourseforum_all_reviews (1).csv")
    parser.add_argument("--output", type=Path, default=root / "data/embeddings_cleaned_new.npz")
    parser.add_argument("--check-only", action="store_true", help="Check cleaning and splits without generating embeddings")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing embedding export")
    args = parser.parse_args()
    # preserve the selected model's current input unless replacement is explicitly requested.
    if not args.check_only and args.output.exists() and not args.overwrite:
        parser.error("Output already exists. Choose --output with a new name, or use --overwrite intentionally.")
    if not args.check_only and args.output.suffix.lower() != ".npz":
        parser.error("The output filename must end in .npz.")
    splits = prepare_splits(args.input)
    if not args.check_only:
        save_embeddings(splits, args.output)


# importing this file makes its functions available; running it starts the local preparation steps.
if __name__ == "__main__":
    main()
