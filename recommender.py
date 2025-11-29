from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .lyrics_features import load_tfidf_for_inference


class SpectralNeighborRecommender:
    """
    Extended content-based recommender using:
    - Audio spectral features (existing)
    - Lyrics TF–IDF vectors (new, optional)
    """

    def __init__(
        self,
        features_audio: pd.DataFrame,
        audio_feature_cols: List[str],
        use_lyrics: bool = True,
        alpha_audio: float = 0.7,
    ):
        """
        features_audio: DataFrame with at least ['track_id'] + audio features
        audio_feature_cols: columns in features_audio to use as audio vector
        use_lyrics: whether to include lyrics in similarity
        alpha_audio: weight for audio similarity (lyrics weight = 1 - alpha_audio)
        """
        self.features_audio = features_audio.copy()
        self.audio_feature_cols = audio_feature_cols
        self.use_lyrics = use_lyrics
        self.alpha_audio = alpha_audio

        # Build index: track_id -> row index
        self.track_ids = self.features_audio["track_id"].astype(str).tolist()
        self.index_by_id: Dict[str, int] = {
            tid: i for i, tid in enumerate(self.track_ids)
        }

        # Build audio feature matrix (numpy)
        self.audio_matrix = self.features_audio[audio_feature_cols].to_numpy().astype(
            np.float32
        )

        # Normalize audio vectors (L2) for cosine similarity
        self.audio_matrix = self._l2_normalize(self.audio_matrix)

        # Load lyrics TF–IDF 
        self.lyrics_matrix = None
        self.lyrics_track_ids = None
        self.lyrics_index_by_id = {}

        if use_lyrics:
            try:
                (
                    tfidf_matrix,
                    _vectorizer,
                    lyrics_track_ids,
                ) = load_tfidf_for_inference()
                self.lyrics_matrix = tfidf_matrix  # sparse matrix
                self.lyrics_track_ids = lyrics_track_ids
                self.lyrics_index_by_id = {
                    tid: i for i, tid in enumerate(self.lyrics_track_ids)
                }
            except FileNotFoundError:
                print("[WARN] Lyrics TF–IDF not found; continuing with audio only.")
                self.use_lyrics = False

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
   
    # Core recommendation method


    def recommend_by_track_id(
        self,
        track_id: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of top-k similar tracks for the given seed track.
        Includes both audio similarity and combined similarity if lyrics are used.
        """
        track_id = str(track_id)
        if track_id not in self.index_by_id:
            raise KeyError(f"Unknown track_id: {track_id}")

        seed_idx = self.index_by_id[track_id]

        # 1. Audio similarity
        seed_audio_vec = self.audio_matrix[seed_idx].reshape(1, -1)
        audio_sim = cosine_similarity(seed_audio_vec, self.audio_matrix)[0]

        # 2. Lyrics similarity 
        if self.use_lyrics and self.lyrics_matrix is not None:
            if track_id in self.lyrics_index_by_id:
                seed_lyrics_idx = self.lyrics_index_by_id[track_id]
                seed_lyrics_vec = self.lyrics_matrix[seed_lyrics_idx]
                # seed_lyrics_vec is 1 x D sparse; use matrix multiplication
                lyrics_sim = cosine_similarity(seed_lyrics_vec, self.lyrics_matrix)[0]
            else:
                # Seed track has no lyrics; fallback = zeros
                lyrics_sim = np.zeros_like(audio_sim)
        else:
            lyrics_sim = np.zeros_like(audio_sim)

        # 3. Combine similarities (late fusion)
        alpha = self.alpha_audio
        combined_sim = alpha * audio_sim + (1.0 - alpha) * lyrics_sim

        # 4. Exclude the seed track itself
        combined_sim[seed_idx] = -np.inf

        # 5. Get top-k indices
        topk_idx = np.argpartition(-combined_sim, k)[:k]
        topk_idx = topk_idx[np.argsort(-combined_sim[topk_idx])]

        rows: List[Dict] = []
        for idx in topk_idx:
            row = self.features_audio.iloc[idx]
            rows.append(
                {
                    "seed_track_id": track_id,
                    "rec_track_id": row["track_id"],
                    "rec_title": row.get("title", None),
                    "rec_artist": row.get("artist", None),
                    "audio_similarity": float(audio_sim[idx]),
                    "lyrics_similarity": float(lyrics_sim[idx]),
                    "combined_similarity": float(combined_sim[idx]),
                }
            )

        return pd.DataFrame(rows)
