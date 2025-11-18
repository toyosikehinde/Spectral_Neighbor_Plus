import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_neighbors(feat_df, track_id, topk=10):
    idx = feat_df.index[feat_df["track_id"] == track_id][0]
    seed = feat_df.iloc[idx].values.reshape(1, -1)

    sims = cosine_similarity(seed, feat_df.values[:, 1:])[0]
    top_idx = sims.argsort()[::-1][1:topk+1]

    return feat_df.iloc[top_idx]
