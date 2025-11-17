from typing import Dict

import numpy as np
import librosa


def _spectral_entropy(S: np.ndarray, eps: float = 1e-10) -> float:
    """
    Entropie spectrale moyenne (en bits) à partir d'un spectrogramme de magnitudes S.

    S : np.ndarray
        Matrice (freq_bins, frames) de magnitudes.
    """
    if S.size == 0:
        return float("nan")

    # Normalisation par frame pour obtenir une distribution de probas
    S = S.astype(np.float64)
    power = S ** 2
    power_sum = np.sum(power, axis=0, keepdims=True) + eps
    p = power / power_sum

    # entropie par frame (base 2)
    entropy = -np.sum(p * np.log2(p + eps), axis=0)
    return float(np.mean(entropy))


def compute_spectral_stats(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Dict[str, float]:
    """
    Calcule plusieurs descripteurs spectraux de base sur un segment audio
    et renvoie des statistiques agrégées (moyenne / écart-type).

    Features :
        - spectral centroid
        - spectral bandwidth
        - spectral rolloff (95%)
        - spectral flatness
        - spectral entropy (moyenne)

    Paramètres
    ----------
    y : np.ndarray
        Signal audio mono (segment).
    sr : int
        Fréquence d'échantillonnage.
    n_fft : int
        Taille de la FFT.
    hop_length : int
        Décalage entre trames.

    Retour
    ------
    stats : Dict[str, float]
        Dictionnaire avec les statistiques des features.
    """
    stats: Dict[str, float] = {}

    if y.size == 0:
        return stats

    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
    )

    # Centroid & bande passante
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

    # Rolloff 95%
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]

    # Flatness
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Entropie spectrale moyenne
    entropy = _spectral_entropy(S)

    def add_basic_stats(name: str, x: np.ndarray) -> None:
        stats[f"{name}_mean"] = float(np.mean(x))
        stats[f"{name}_std"] = float(np.std(x))

    add_basic_stats("spec_centroid_hz", centroid)
    add_basic_stats("spec_bandwidth_hz", bandwidth)
    add_basic_stats("spec_rolloff95_hz", rolloff)
    add_basic_stats("spec_flatness", flatness)

    stats["spec_entropy_bits_mean"] = float(entropy)

    return stats
