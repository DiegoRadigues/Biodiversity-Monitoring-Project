import sys
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# === Préparer les imports du package src ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # type: ignore
    DATA_PROCESSED,
    TARGET_SR,
    STFT_N_FFT,
    STFT_HOP_LENGTH,
    STFT_WINDOW,
)
from src.signal.stft import compute_spectrogram  # type: ignore

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp1"
SEGMENTS_CSV = EXPERIMENT_DIR / "segments.csv"
OUT_DIR = PROJECT_ROOT / "assets" / "figures_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_filtered_audio(rel_file: str) -> tuple[np.ndarray, int]:
    """
    rel_file : chemin relatif du fichier brut, ex. 'Bird songs/merle.wav'
    -> charge data/processed/Bird songs/merle_bandpass.wav
    """
    rel_path = Path(rel_file)
    filtered_name = rel_path.stem + "_bandpass.wav"
    filtered_path = DATA_PROCESSED / rel_path.parent / filtered_name

    y, sr = librosa.load(filtered_path, sr=TARGET_SR, mono=True)
    return y.astype(np.float32), sr


def plot_annotated_spectrogram(rel_file: str, df_file: pd.DataFrame) -> None:
    """
    Génère un spectrogramme avec les segments surlignés en vertical.
    """
    print(f"- Annotation : {rel_file}")

    y, sr = load_filtered_audio(rel_file)

    S_db, freqs, times = compute_spectrogram(
        y,
        sr=sr,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        window=STFT_WINDOW,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=STFT_HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Spectrogramme annoté — {rel_file}")

    # Dessiner les segments (zones verticales)
    for _, row in df_file.iterrows():
        t_on = float(row["t_onset_s"])
        t_off = float(row["t_offset_s"])
        ax.axvspan(t_on, t_off, alpha=0.25)

    out_name = Path(rel_file).stem + "_segments.png"
    out_path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  -> figure annotée sauvegardée : {out_path}")


def main():
    if not SEGMENTS_CSV.exists():
        print(f"segments.csv introuvable : {SEGMENTS_CSV}")
        print("Lance d'abord : python scripts/run_pipeline.py")
        return

    df = pd.read_csv(SEGMENTS_CSV)
    if df.empty:
        print("segments.csv est vide.")
        return

    # Stats rapides
    print("\n=== Stats segments ===")
    counts = df.groupby("file")["segment_id"].count()
    print(counts)

    # Figures annotées pour chaque fichier
    print("\n=== Génération des spectrogrammes annotés ===")
    for rel_file, df_file in df.groupby("file"):
        plot_annotated_spectrogram(rel_file, df_file)


if __name__ == "__main__":
    main()
