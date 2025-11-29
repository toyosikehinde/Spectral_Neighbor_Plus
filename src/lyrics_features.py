import re
import string
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

LYRICS_RAW_CSV = DATA_RAW / "lyrics_raw.csv"
TFIDF_MATRIX_PATH = DATA_PROCESSED / "lyrics_tfidf.npz"
TFIDF_VECTORIZER_PATH = DATA_PROCESSED / "lyrics_tfidf_vectorizer.joblib"
LYRICS_TRACK_IDS_PATH = DATA_PROCESSED / "lyrics_track_ids.csv"

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def basic_clean(text: str) -> str:
    """Minimal cleaning for lyrics."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\[.*?\]", " ", text)      # remove [chorus] etc
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lyrics_dataframe(path: Path = LYRICS_RAW_CSV) -> pd.DataFrame:
    """Load raw lyrics and create a cleaned column."""
    df = pd.read_csv(path)
    if "track_id" not in df.columns or "lyrics" not in df.columns:
        raise ValueError("lyrics_raw.csv must contain 'track_id' and 'lyrics'.")
    df = df.dropna(subset=["lyrics"]).copy()
    df["track_id"] = df["track_id"].astype(str)
    df["lyrics_clean"] = df["lyrics"].apply(basic_clean)
    return df


def build_tfidf_vectorizer(
    lyrics_texts: pd.Series,
    min_df: int = 3,
    max_df: float = 0.7,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Fit TF–IDF on cleaned lyrics."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    vectorizer.fit(lyrics_texts)
    return vectorizer


def build_and_save_tfidf() -> None:
    """End-to-end build and save TF–IDF artifacts."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    lyrics_df = load_lyrics_dataframe(LYRICS_RAW_CSV)
    texts = lyrics_df["lyrics_clean"].astype(str).tolist()
    track_ids = lyrics_df["track_id"].astype(str).tolist()

    vectorizer = build_tfidf_vectorizer(lyrics_df["lyrics_clean"])
    tfidf_matrix = vectorizer.transform(texts)   # L2-normalized sparse matrix

    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    pd.DataFrame({"track_id": track_ids}).to_csv(LYRICS_TRACK_IDS_PATH, index=False)

    print(f"Saved TF–IDF matrix to {TFIDF_MATRIX_PATH}")
    print(f"Saved vectorizer to {TFIDF_VECTORIZER_PATH}")
    print(f"Saved track_id mapping to {LYRICS_TRACK_IDS_PATH}")


def load_tfidf_for_inference():
    """Helper used by neighbor features."""
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    ids_df = pd.read_csv(LYRICS_TRACK_IDS_PATH)
    track_ids = ids_df["track_id"].astype(str).tolist()
    return tfidf_matrix, vectorizer, track_ids


if __name__ == "__main__":
    build_and_save_tfidf()
