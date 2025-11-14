from typing import Tuple

import numpy as np
from scipy.signal import firwin, filtfilt


def design_fir_bandpass(
    fs: int,
    lowcut: float,
    highcut: float,
    numtaps: int = 1025,
) -> np.ndarray:
    """
    Conçoit un filtre passe-bande FIR à phase linéaire entre lowcut et highcut (Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    taps = firwin(
        numtaps=numtaps,
        cutoff=[low, high],
        pass_zero=False,
        window="hann",
    )
    return taps


def apply_fir_bandpass(
    y: np.ndarray,
    fs: int,
    lowcut: float,
    highcut: float,
    numtaps: int = 1025,
) -> np.ndarray:
    """
    Applique le filtre passe-bande FIR avec filtfilt (zéro phase).
    """
    taps = design_fir_bandpass(fs, lowcut, highcut, numtaps)

    # filtfilt = filtrage aller-retour, pas de déphasage
    y_filt = filtfilt(taps, [1.0], y)

    return y_filt.astype(np.float32)
