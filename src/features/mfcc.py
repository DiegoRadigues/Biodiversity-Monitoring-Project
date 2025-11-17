from typing import Dict

import numpy as np
import librosa


def compute_mfcc_stats(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Dict[str, float]:
    """
    Calcule des MFCC sur un segment audio et renvoie des statistiques
    (moyenne, écart-type, min, max) pour chaque coefficient.

    Les clés retournées sont du type :
        - "mfcc_01_mean", "mfcc_01_std", "mfcc_01_min", "mfcc_01_max", etc.

    Paramètres
    ----------
    y : np.ndarray
        Signal audio mono (segment).
    sr : int
        Fréquence d'échantillonnage.
    n_mfcc : int
        Nombre de coefficients MFCC à calculer.
    n_fft : int
        Taille de la FFT (en échantillons).
    hop_length : int
        Décalage entre deux trames (en échantillons).

    Retour
    ------
    stats : Dict[str, float]
        Dictionnaire de statistiques agrégées sur les MFCC.
    """
    stats: Dict[str, float] = {}

    if y.size == 0:
        return stats

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # shape : (n_mfcc, frames)

    for i in range(n_mfcc):
        coef = mfcc[i, :]
        prefix = f"mfcc_{i+1:02d}"
        stats[f"{prefix}_mean"] = float(np.mean(coef))
        stats[f"{prefix}_std"] = float(np.std(coef))
        stats[f"{prefix}_min"] = float(np.min(coef))
        stats[f"{prefix}_max"] = float(np.max(coef))

    return stats
