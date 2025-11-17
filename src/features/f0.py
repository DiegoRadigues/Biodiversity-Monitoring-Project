from typing import Dict

import numpy as np
import librosa


def compute_f0_stats(
    y: np.ndarray,
    sr: int,
    fmin: float = 300.0,
    fmax: float = 8000.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Dict[str, float]:
    """
    Estime la fréquence fondamentale f0 sur un segment audio à l'aide de YIN,
    puis calcule des statistiques globales.

    Paramètres
    ----------
    y : np.ndarray
        Signal audio mono (segment).
    sr : int
        Fréquence d'échantillonnage.
    fmin : float
        Fréquence minimale attendue (Hz).
    fmax : float
        Fréquence maximale attendue (Hz).
    frame_length : int
        Longueur de trame pour YIN.
    hop_length : int
        Décalage entre trames.

    Retour
    ------
    stats : Dict[str, float]
        Clés :
            - f0_mean_hz
            - f0_std_hz
            - f0_min_hz
            - f0_max_hz
            - f0_voiced_ratio
    """
    stats: Dict[str, float] = {}

    if y.size == 0:
        return stats

    # YIN renvoie un f0 par frame
    try:
        f0 = librosa.yin(
            y=y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        # En cas de problème numérique, on renvoie des NaN
        stats["f0_mean_hz"] = float("nan")
        stats["f0_std_hz"] = float("nan")
        stats["f0_min_hz"] = float("nan")
        stats["f0_max_hz"] = float("nan")
        stats["f0_voiced_ratio"] = 0.0
        return stats

    f0 = np.asarray(f0, dtype=np.float64)

    # Considérer comme non-voisé les valeurs <= 0 ou NaN
    voiced_mask = np.isfinite(f0) & (f0 > 0.0)

    if not np.any(voiced_mask):
        stats["f0_mean_hz"] = float("nan")
        stats["f0_std_hz"] = float("nan")
        stats["f0_min_hz"] = float("nan")
        stats["f0_max_hz"] = float("nan")
        stats["f0_voiced_ratio"] = 0.0
        return stats

    f0_voiced = f0[voiced_mask]

    stats["f0_mean_hz"] = float(np.mean(f0_voiced))
    stats["f0_std_hz"] = float(np.std(f0_voiced))
    stats["f0_min_hz"] = float(np.min(f0_voiced))
    stats["f0_max_hz"] = float(np.max(f0_voiced))
    stats["f0_voiced_ratio"] = float(f0_voiced.size / f0.size)

    return stats
