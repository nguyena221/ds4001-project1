"""prepare cleaned, non-chunked MiniLM embeddings using Zilan's split procedure."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_review(text):
    """remove the same webpage headers and footers as the selected notebook version."""
    # remove the term, year, and average rating from the webpage heading.
    text = re.sub(
        r'^(Fall|Spring|Summer|Winter|January)\s+\d{4}\s+\d+(?:\.\d+)?\s+AVERAGE\s*',
        '', text, flags=re.IGNORECASE
    )
    # remove the vote count and date at the end, keeping the student's review.
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
    # stop if anything is missing so we don't accidentally change the review groups.
    if reviews[list(required)].isna().any().any():
        raise ValueError("Missing reviews or ratings: resolve these before reproducing the split.")
    if not reviews["review_text"].map(lambda text: isinstance(text, str)).all():
        raise ValueError("Every review must be text.")
    if not reviews["difficulty_rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Difficulty ratings must be whole numbers from 1 to 5.")

    reviews["cleaned_review"] = reviews["review_text"].apply(clean_review)
    if reviews["cleaned_review"].eq("").any():
        raise ValueError("Cleaning produced an empty review; inspect the input first.")
    X = reviews["cleaned_review"]
    y = reviews["difficulty_rating"]

    # use 70 percent for training and keep a similar mix of ratings in each group.
    # the seed of 42 repeats the split when the data and row order stay the same.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # split the rest equally into validation and test groups, with 15 percent in each.
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
    from sentence_transformers import SentenceTransformer

    # use the already trained MiniLM model to turn the reviews into numbers.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    arrays = {}
    for name, (texts, labels) in splits.items():
        # encode each review without chunking. text beyond the model's length limit gets cut off.
        features = model.encode(texts.tolist(), show_progress_bar=True)
        if features.shape != (len(labels), 384) or not np.isfinite(features).all():
            raise ValueError(f"Unexpected embedding shape or values for {name}.")
        arrays[f"X_{name}"] = features
        arrays[f"y_{name}"] = labels.to_numpy()
        print(f"{name} embeddings: {features.shape}; labels: {labels.shape}")
    # save the review numbers and ratings in the same order so they stay matched.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"Saved private embeddings to {output_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/thecourseforum_all_reviews (1).csv")
    parser.add_argument("--output", type=Path, default=root / "data/embeddings_cleaned_new.npz")
    parser.add_argument("--check-only", action="store_true", help="Check cleaning and splits without generating embeddings")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing embedding export")
    args = parser.parse_args()
    # protect the existing file unless we specifically choose to overwrite it.
    if not args.check_only and args.output.exists() and not args.overwrite:
        parser.error("Output already exists. Choose --output with a new name, or use --overwrite intentionally.")
    if not args.check_only and args.output.suffix.lower() != ".npz":
        parser.error("The output filename must end in .npz.")
    splits = prepare_splits(args.input)
    if not args.check_only:
        save_embeddings(splits, args.output)


if __name__ == "__main__":
    main()
