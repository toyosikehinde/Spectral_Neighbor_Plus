import librosa
import numpy as np
import pandas as pd
import os

def extract_features(path):
    y, sr = librosa.load(path, sr=22050)

    # STFT and Mel
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel)

    # MFCC and deltas
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Texture features
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onset_density = np.mean(onset_env)

    # Aggregate features
    def stats(X):
        return np.hstack([np.mean(X, axis=1), np.std(X, axis=1)])

    feature_vec = np.concatenate([
        stats(mfcc), stats(delta), stats(delta2),
        stats(centroid), stats(rolloff), stats(flatness),
        stats(zcr), stats(contrast), stats(bandwidth),
        np.array([tempo, onset_density])
    ])

    return feature_vec
