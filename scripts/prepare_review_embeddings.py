"""prepare cleaned, non-chunked MiniLM embeddings using Zilan's split procedure."""

# these help us find the files, choose how to run the script, and clean up the scraped text.
import argparse
from pathlib import Path
import re

# pandas opens our data file, numpy saves the review numbers, and scikit-learn splits up the reviews.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_review(text):
    """remove the same webpage headers and footers as the selected notebook version."""
    # remove the webpage stuff before the actual review, like the term, year, and average rating.
    # january was getting missed before, so it's included here too.
    text = re.sub(
        r'^(Fall|Spring|Summer|Winter|January)\s+\d{4}\s+\d+(?:\.\d+)?\s+AVERAGE\s*',
        '', text, flags=re.IGNORECASE
    )
    # remove the vote count and posting date at the end. leave the student's writing in between.
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
    # stop if something is missing instead of quietly dropping reviews and changing our split.
    # our current dataset has all the review texts and difficulty ratings, so this should pass.
    if reviews[list(required)].isna().any().any():
        raise ValueError("Missing reviews or ratings: resolve these before reproducing the split.")
    if not reviews["review_text"].map(lambda text: isinstance(text, str)).all():
        raise ValueError("Every review must be text.")
    if not reviews["difficulty_rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Difficulty ratings must be whole numbers from 1 to 5.")

    # clean the reviews once with the rule above. the original notebook had two versions of this.
    # we're not adding new changes to capitalization, punctuation, or markdown in this version.
    reviews["cleaned_review"] = reviews["review_text"].apply(clean_review)
    if reviews["cleaned_review"].eq("").any():
        raise ValueError("Cleaning produced an empty review; inspect the input first.")
    X = reviews["cleaned_review"]
    y = reviews["difficulty_rating"]

    # put 70 percent of the reviews into the training group and leave 30 percent aside for now.
    # keep a similar mix of difficulty ratings in each group.
    # using the same random seed of 42 repeats the split when the data and row order stay the same.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # split the remaining reviews in half, giving us 15 percent for validation and 15 percent for testing.
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
    # only load this package when we're ready to make the embeddings.
    # that way, check-only can check the cleaning and splits without downloading minilm.
    from sentence_transformers import SentenceTransformer

    # minilm is already trained. we're using it to turn our reviews into numbers, not training it again.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    arrays = {}
    for name, (texts, labels) in splits.items():
        # process the training, validation, and test reviews the same way.
        # we're keeping each review together instead of splitting it into smaller pieces.
        # this means minilm still cuts off text that goes past its normal length limit.
        features = model.encode(texts.tolist(), show_progress_bar=True)
        if features.shape != (len(labels), 384) or not np.isfinite(features).all():
            raise ValueError(f"Unexpected embedding shape or values for {name}.")
        arrays[f"X_{name}"] = features
        arrays[f"y_{name}"] = labels.to_numpy()
        print(f"{name} embeddings: {features.shape}; labels: {labels.shape}")
    # keep the order the same so the first review's numbers go with the first rating, and so on.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"Saved private embeddings to {output_path}")


def main():
    # look in our project's data folder by default, even if the terminal is somewhere else.
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/thecourseforum_all_reviews (1).csv")
    parser.add_argument("--output", type=Path, default=root / "data/embeddings_split.npz")
    parser.add_argument("--check-only", action="store_true", help="Check cleaning and splits without generating embeddings")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing embedding export")
    args = parser.parse_args()
    # don't accidentally replace the embeddings we're already using. overwriting has to be intentional.
    if not args.check_only and args.output.exists() and not args.overwrite:
        parser.error("Output already exists. Choose --output with a new name, or use --overwrite intentionally.")
    if not args.check_only and args.output.suffix.lower() != ".npz":
        parser.error("The output filename must end in .npz.")
    splits = prepare_splits(args.input)
    if not args.check_only:
        save_embeddings(splits, args.output)


# start the steps when we run this file, but not when another script just imports its functions.
if __name__ == "__main__":
    main()
