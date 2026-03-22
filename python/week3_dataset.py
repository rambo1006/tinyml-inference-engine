import os
import numpy as np
import librosa
import urllib.request
import tarfile
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "/Users/sharansai/Desktop/audio-event-detection/data"
KEYWORDS     = ["yes", "no", "stop"]
SAMPLE_RATE  = 16000
DURATION     = 1.0
N_MFCC       = 13


def audio_to_mfcc(filepath):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return mfcc.mean(axis=1).astype(np.float32)


def build_dataset():
    X, y = [], []

    for label, keyword in enumerate(KEYWORDS):
        folder = os.path.join(DATA_DIR, keyword)
        files  = list(Path(folder).glob("*.wav"))[:500]
        print(f"Processing '{keyword}': {len(files)} files...")

        for filepath in files:
            try:
                features = audio_to_mfcc(str(filepath))
                X.append(features)
                y.append(label)
            except Exception:
                pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split     = int(0.8 * len(X))
    X_train   = X[:split]
    y_train   = y[:split]
    X_val     = X[split:]
    y_val     = y[split:]

    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy",   X_val)
    np.save("data/y_val.npy",   y_val)

    print(f"\nDataset built:")
    print(f"  Training   : {len(X_train)} samples")
    print(f"  Validation : {len(X_val)} samples")
    print(f"  Features   : {X_train.shape[1]} MFCC coefficients per sample")
    print(f"  Classes    : {KEYWORDS}")
    print(f"\nFiles saved to data/")
    return X_train, y_train, X_val, y_val


def visualize_mfccs(X_train, y_train):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Average MFCC features per keyword", fontsize=13)

    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    for i, (keyword, color) in enumerate(zip(KEYWORDS, colors)):
        samples = X_train[y_train == i]
        mean    = samples.mean(axis=0)
        std     = samples.std(axis=0)

        axes[i].bar(range(N_MFCC), mean, color=color, alpha=0.8, label="mean")
        axes[i].fill_between(range(N_MFCC),
                             mean - std, mean + std,
                             alpha=0.2, color=color, label="±1 std")
        axes[i].set_title(f'"{keyword}"')
        axes[i].set_xlabel("MFCC coefficient index")
        axes[i].set_ylabel("Value")
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/mfcc_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved to results/mfcc_visualization.png")


X_train, y_train, X_val, y_val = build_dataset()
visualize_mfccs(X_train, y_train)

print("\nNext: run week4_train.py to train the PyTorch model on this data.")