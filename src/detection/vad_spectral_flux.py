import numpy as np
import librosa
from typing import List, Tuple


def _compute_spectral_flux(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Calcule le flux spectral positif frame par frame.
    """
    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
    )  # shape: (freq_bins, frames)

    # Différence entre frames successives
    diff = np.diff(S, axis=1)
    # On ne garde que les augmentations (partie positive)
    diff_pos = np.maximum(diff, 0.0)

    # Flux spectral = norme des différences positives
    flux = np.sqrt((diff_pos ** 2).sum(axis=0))  # shape: (frames-1,)

    # On insère un 0 au début pour aligner avec les frames STFT
    flux = np.concatenate([[0.0], flux])

    return flux


def _binarize_flux(
    flux: np.ndarray,
    k: float = 2.5,
    min_duration_s: float = 0.15,
    hop_length: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """
    Seuillage adaptatif simple sur le flux spectral.

    - seuil = médiane + k * écart-type
    - on enlève les segments trop courts (< min_duration_s)
    """
    if flux.size == 0:
        return np.zeros_like(flux, dtype=bool)

    # Normalisation grossière
    mu = np.median(flux)
    sigma = np.std(flux)
    if sigma < 1e-8:
        sigma = 1e-8

    thresh = mu + k * sigma
    active = flux > thresh  # bool array

    # Suppression des segments trop courts
    min_frames = int(np.round(min_duration_s * sr / hop_length))
    if min_frames <= 1:
        return active

    # On garde les runs de True suffisamment longs
    cleaned = np.zeros_like(active, dtype=bool)
    start = None
    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if (end - start) >= min_frames:
                cleaned[start:end] = True
            start = None
    # Fin de séquence
    if start is not None:
        end = len(active)
        if (end - start) >= min_frames:
            cleaned[start:end] = True

    return cleaned


def activity_to_segments(
    activity: np.ndarray,
    sr: int,
    hop_length: int,
) -> List[Tuple[float, float]]:
    """
    Transforme un masque booléen frame par frame en liste de segments (t_onset, t_offset).
    """
    segments: List[Tuple[float, float]] = []
    if activity.size == 0:
        return segments

    start = None
    for i, val in enumerate(activity):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            t_on = start * hop_length / sr
            t_off = end * hop_length / sr
            segments.append((t_on, t_off))
            start = None

    if start is not None:
        end = len(activity)
        t_on = start * hop_length / sr
        t_off = end * hop_length / sr
        segments.append((t_on, t_off))

    return segments


def detect_activity(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    k: float = 2.5,
    min_duration_s: float = 0.15,
) -> List[Tuple[float, float]]:
    """
    Pipeline complet :
    - calcule le flux spectral
    - applique un seuillage adaptatif
    - renvoie une liste de segments (t_onset, t_offset) en secondes.
    """
    flux = _compute_spectral_flux(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    activity = _binarize_flux(
        flux,
        k=k,
        min_duration_s=min_duration_s,
        hop_length=hop_length,
        sr=sr,
    )
    segments = activity_to_segments(activity, sr=sr, hop_length=hop_length)
    return segments
