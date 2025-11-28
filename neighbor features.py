# neighbor features.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from lyrics_features import load_tfidf_for_inference

# load your audio features DataFrame (whatever you do now)
audio_df = pd.read_csv("data/processed/audio_features.csv")   # adapt to your filename
audio_df["track_id"] = audio_df["track_id"].astype(str)

AUDIO_FEATURE_COLS = [
    # the columns you already use as your feature vector
    # "mfcc_01_mean", "mfcc_01_std", ...
]

audio_matrix = audio_df[AUDIO_FEATURE_COLS].to_numpy().astype(np.float32)

# L2 normalize for cosine similarity
audio_norms = np.linalg.norm(audio_matrix, axis=1, keepdims=True)
audio_norms[audio_norms == 0] = 1.0
audio_matrix = audio_matrix / audio_norms

INDEX_BY_ID = {tid: i for i, tid in enumerate(audio_df["track_id"].tolist())}

# Load lyrics TF–IDF (if available)
try:
    lyrics_matrix, _vectorizer, lyrics_track_ids = load_tfidf_for_inference()
    LYRICS_INDEX_BY_ID = {tid: i for i, tid in enumerate(lyrics_track_ids)}
    USE_LYRICS = True
except FileNotFoundError:
    print("Lyrics TF–IDF not found, falling back to audio-only.")
    lyrics_matrix = None
    LYRICS_INDEX_BY_ID = {}
    USE_LYRICS = False
