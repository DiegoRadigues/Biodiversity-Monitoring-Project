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


def merge_segments(
    segments: List[Tuple[float, float]],
    min_gap_s: float = 0.10,
    min_duration_s: float = 0.08,
) -> List[Tuple[float, float]]:
    """
    Regroupe les segments trop proches dans le temps.

    - Si l'écart entre la fin d'un segment et le début du suivant
      est < min_gap_s, on les fusionne.
    - Après fusion, on supprime les segments dont la durée < min_duration_s.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])

    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = segments[0]

    for start, end in segments[1:]:
        # si le prochain segment est très proche du courant -> fusion
        if start - cur_end <= min_gap_s:
            cur_end = max(cur_end, end)
        else:
            if (cur_end - cur_start) >= min_duration_s:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    # dernier segment
    if (cur_end - cur_start) >= min_duration_s:
        merged.append((cur_start, cur_end))

    return merged


def detect_activity(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    k: float = 0.8,
    min_duration_s: float = 0.08,
    raw_min_duration_s: float = 0.03,
    merge_gap_s: float = 0.10,
) -> List[Tuple[float, float]]:
    """
    Pipeline complet :
    - calcule le flux spectral
    - applique un seuillage adaptatif (très sensible)
    - regroupe les segments proches dans le temps
    - renvoie une liste de segments (t_onset, t_offset) en secondes.

    min_duration_s : durée minimale après fusion
    raw_min_duration_s : durée minimale avant fusion (très petite)
    """
    # 1) VAD très sensible (petite durée minimale brute)
    flux = _compute_spectral_flux(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    activity = _binarize_flux(
        flux,
        k=k,
        min_duration_s=raw_min_duration_s,
        hop_length=hop_length,
        sr=sr,
    )

    # 2) Conversion en segments bruts
    segments = activity_to_segments(activity, sr=sr, hop_length=hop_length)

    # 3) Regroupement temporel
    segments_merged = merge_segments(
        segments,
        min_gap_s=merge_gap_s,
        min_duration_s=min_duration_s,
    )

    return segments_merged

def detect_segments_vad(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    threshold: float = 0.8,          # joue le rôle de k
    smooth_win_s: float = 0.20,      # gardé pour compatibilité, pas utilisé ici
    min_syllable_duration_s: float = 0.08,
    merge_gap_s: float = 0.10,
    **kwargs,
):
    """
    Wrapper pour compatibilité avec l'ancienne API `detect_segments_vad`.

    Paramètres compatibles avec l'ancien code :
    - threshold : facteur sur l'écart-type (équivalent de k dans `_binarize_flux`)
    - smooth_win_s : gardé mais non utilisé ici
    - min_syllable_duration_s : durée minimale d'une syllabe (utilisée comme min_duration_s)
    - merge_gap_s : temps max entre segments pour les fusionner

    `**kwargs` permet d'ignorer proprement d'autres arguments éventuels
    passés par du vieux code.
    """
    return detect_activity(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        k=threshold,
        # on utilise min_syllable_duration_s comme durée minimale finale
        min_duration_s=min_syllable_duration_s,
        # durée minimale brute avant fusion : un peu plus courte
        raw_min_duration_s=min_syllable_duration_s * 0.5,
        merge_gap_s=merge_gap_s,
    )



  

