from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa


def load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Charge un fichier audio, convertit en mono, resample à target_sr si besoin
    et normalise l'amplitude entre -1 et 1.

    Retourne:
        y (np.ndarray): signal mono
        sr (int): fréquence d'échantillonnage (target_sr)
    """
    path = Path(path)

    # Lecture avec soundfile (préserve le sr d'origine)
    y, sr = sf.read(path, always_2d=False)

    # Conversion en mono si nécessaire
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    # Normalisation (éviter division par zéro)
    max_abs = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    y = y / max_abs

    # Resample si fréquence différente
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y.astype(np.float32), sr
