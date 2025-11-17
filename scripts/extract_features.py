import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import librosa

# === Préparer les imports du package src ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # type: ignore
    DATA_PROCESSED,
    TARGET_SR,
    STFT_N_FFT,
    STFT_HOP_LENGTH,
)
from src.features.mfcc import compute_mfcc_stats  # type: ignore
from src.features.spectral_stats import compute_spectral_stats  # type: ignore
from src.features.f0 import compute_f0_stats  # type: ignore


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp1"
SEGMENTS_CSV = EXPERIMENT_DIR / "segments.csv"
FEATURES_CSV = EXPERIMENT_DIR / "features.csv"


def load_filtered_audio(rel_file: str) -> tuple[np.ndarray, int]:
    """
    Charge l'audio filtré correspondant à un fichier brut.

    rel_file : chemin relatif du fichier brut (ex. 'Bird songs/merle.wav')

    -> charge data/processed/Bird songs/merle_bandpass.wav
    """
    rel_path = Path(rel_file)
    filtered_name = rel_path.stem + "_bandpass.wav"
    filtered_path = DATA_PROCESSED / rel_path.parent / filtered_name

    if not filtered_path.exists():
        raise FileNotFoundError(f"Fichier audio filtré introuvable : {filtered_path}")

    y, sr = librosa.load(filtered_path, sr=TARGET_SR, mono=True)
    return y.astype(np.float32), sr


def extract_features_for_segment(
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """
    Calcule toutes les features pour un segment (MFCC + spectrales + f0)
    et renvoie un dictionnaire plat.
    """
    features: Dict[str, Any] = {}

    # MFCC
    mfcc_stats = compute_mfcc_stats(
        y,
        sr=sr,
        n_mfcc=20,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
    )
    features.update(mfcc_stats)

    # Statistiques spectrales
    spec_stats = compute_spectral_stats(
        y,
        sr=sr,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
    )
    features.update(spec_stats)

    # f0
    f0_stats = compute_f0_stats(
        y,
        sr=sr,
        fmin=300.0,
        fmax=8000.0,
        frame_length=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
    )
    features.update(f0_stats)

    return features


def main():
    if not SEGMENTS_CSV.exists():
        print(f"segments.csv introuvable : {SEGMENTS_CSV}")
        print("Lance d'abord : python scripts/run_pipeline.py")
        return

    df_segments = pd.read_csv(SEGMENTS_CSV)
    if df_segments.empty:
        print("segments.csv est vide, rien à extraire.")
        return

    print(f"{len(df_segments)} segments trouvés dans {SEGMENTS_CSV}")

    all_rows: List[Dict[str, Any]] = []

    # On groupe par fichier pour ne charger chaque audio filtré qu'une fois
    for rel_file, df_file in df_segments.groupby("file"):
        print(f"\n=== Fichier : {rel_file} ===")

        try:
            y_full, sr = load_filtered_audio(rel_file)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            continue

        for _, row in df_file.iterrows():
            seg_id = int(row["segment_id"])
            t_on = float(row["t_onset_s"])
            t_off = float(row["t_offset_s"])

            start_sample = int(round(t_on * sr))
            end_sample = int(round(t_off * sr))
            start_sample = max(0, start_sample)
            end_sample = min(len(y_full), max(start_sample + 1, end_sample))

            y_seg = y_full[start_sample:end_sample]
            duration_s = (end_sample - start_sample) / sr

            print(f"  - segment {seg_id} : {t_on:.3f}s -> {t_off:.3f}s (durée ~ {duration_s:.3f}s)")

            base_info: Dict[str, Any] = {
                "file": rel_file,
                "segment_id": seg_id,
                "t_onset_s": t_on,
                "t_offset_s": t_off,
                "duration_s": duration_s,
            }

            feat = extract_features_for_segment(y_seg, sr)
            base_info.update(feat)
            all_rows.append(base_info)

    if not all_rows:
        print("Aucune feature générée.")
        return

    df_feat = pd.DataFrame(all_rows)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(FEATURES_CSV, index=False)

    print(f"\nFeatures sauvegardées dans : {FEATURES_CSV}")
    print(f"Nombre de segments avec features : {len(df_feat)}")
    print(f"Nombre de colonnes : {df_feat.shape[1]}")


if __name__ == "__main__":
    main()
