from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule un spectrogramme en dB.

    Retourne:
        S_db: matrice (freq x temps) en dB
        freqs: vecteur des fréquences (Hz)
        times: vecteur des temps (s)
    """
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
    )
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length
    )

    return S_db, freqs, times


def save_spectrogram_figure(
    S_db: np.ndarray,
    sr: int,
    hop_length: int,
    out_path: Path,
    cmap: str = "magma",
) -> None:
    """
    Sauvegarde un spectrogramme temps-fréquence en image.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap=cmap,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogramme (STFT)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
