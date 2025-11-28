
def recommend_by_track_id(track_id: str, k: int = 10, alpha_audio: float = 0.7) -> pd.DataFrame:
    """
    Return top-k neighbors using late fusion of audio and lyrics.
    alpha_audio controls the weight of audio vs lyrics (0 to 1).
    """
    tid = str(track_id)
    if tid not in INDEX_BY_ID:
        raise KeyError(f"Unknown track_id: {tid}")

    seed_idx = INDEX_BY_ID[tid]

    # 1. Audio similarity
    seed_audio_vec = audio_matrix[seed_idx].reshape(1, -1)
    audio_sim = cosine_similarity(seed_audio_vec, audio_matrix)[0]

    # 2. Lyrics similarity
    if USE_LYRICS and lyrics_matrix is not None and tid in LYRICS_INDEX_BY_ID:
        seed_lyrics_idx = LYRICS_INDEX_BY_ID[tid]
        seed_lyrics_vec = lyrics_matrix[seed_lyrics_idx]
        lyrics_sim = cosine_similarity(seed_lyrics_vec, lyrics_matrix)[0]
    else:
        lyrics_sim = np.zeros_like(audio_sim)

    # 3. Late fusion
    combined_sim = alpha_audio * audio_sim + (1.0 - alpha_audio) * lyrics_sim

    # 4. Exclude seed itself
    combined_sim[seed_idx] = -np.inf

    # 5. Get top-k indices
    topk_idx = np.argpartition(-combined_sim, k)[:k]
    topk_idx = topk_idx[np.argsort(-combined_sim[topk_idx])]

    rows = []
    for idx in topk_idx:
        row = audio_df.iloc[idx]
        rows.append(
            {
                "seed_track_id": tid,
                "rec_track_id": row["track_id"],
                "title": row.get("title", None),
                "artist": row.get("artist", None),
                "audio_similarity": float(audio_sim[idx]),
                "lyrics_similarity": float(lyrics_sim[idx]),
                "combined_similarity": float(combined_sim[idx]),
            }
        )

    return pd.DataFrame(rows)
