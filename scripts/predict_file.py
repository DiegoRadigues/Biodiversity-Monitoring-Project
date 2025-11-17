import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import librosa

# pipeline modules déjà existants
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TARGET_SR,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FIR_NUMTAPS,
    STFT_N_FFT,
    STFT_HOP_LENGTH,
)

from src.signal.fir_bandpass import design_fir_bandpass  # type: ignore
from src.signal.stft import compute_spectrogram  # facultatif
from src.detection.vad_spectral_flux import detect_segments_vad  # type: ignore

from src.features.mfcc import compute_mfcc_stats  # type: ignore
from src.features.spectral_stats import compute_spectral_stats  # type: ignore
from src.features.f0 import compute_f0_stats  # type: ignore


MODEL_PATH = ROOT / "experiments" / "exp1" / "models" / "rf_segments.joblib"


def extract_features_from_segment(y_seg, sr):
    """Pack toutes les features du pipeline dans un dict."""
    feats = {}
    feats.update(
        compute_mfcc_stats(
            y_seg, sr, n_mfcc=20, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH
        )
    )
    feats.update(
        compute_spectral_stats(
            y_seg, sr, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH
        )
    )
    feats.update(
        compute_f0_stats(
            y_seg,
            sr,
            fmin=300.0,
            fmax=8000.0,
            frame_length=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
        )
    )
    return feats


def predict_file(path_wav: str):
    path_wav = Path(path_wav)
    print(f"\n=== Analyse de {path_wav.name} ===")

    # 1. Charger audio
    y, sr = librosa.load(path_wav, sr=TARGET_SR, mono=True)
    print(f"Audio chargé : durée = {len(y)/sr:.2f} s, sr = {sr} Hz")

    # 2. FIR band-pass
    fir = design_fir_bandpass(sr, BANDPASS_LOW, BANDPASS_HIGH, numtaps=FIR_NUMTAPS)
    y_filt = np.convolve(y, fir, mode="same")

    # 3. Détection des segments
    segments = detect_segments_vad(
        y_filt,
        sr,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        smooth_win_s=0.05,
        threshold=0.10,
        min_syllable_duration_s=0.05,
        min_silence_duration_s=0.05,
    )

    print(f"{len(segments)} segments détectés")

    # 4. Charger modèle
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    predictions = []

    # 5. Extraire features par segment
    for i, (t_on, t_off) in enumerate(segments):
        start = int(t_on * sr)
        end = int(t_off * sr)
        y_seg = y_filt[start:end]

        feat = extract_features_from_segment(y_seg, sr)
        df_seg = pd.DataFrame([feat])
        df_seg = df_seg.reindex(columns=feature_cols, fill_value=0.0)

        # 6. Prédiction
        pred = pipeline.predict(df_seg)[0]
        proba = (
            pipeline.predict_proba(df_seg)[0].max()
            if hasattr(pipeline, "predict_proba")
            else None
        )

        print(f"  - segment {i}: {t_on:.2f}s → {t_off:.2f}s → {pred} (conf={proba:.2f})")

        predictions.append((pred, proba, t_on, t_off))

    if not predictions:
        print("Aucun segment : impossible de prédire.")
        return

    # 7. Vote majoritaire
    species_list = [p[0] for p in predictions]
    dominant = pd.Series(species_list).mode()[0]

    print("\n=== Espèce dominante prédite ===")
    print(dominant)

    return {
        "segments": predictions,
        "dominant_species": dominant,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/predict_file.py <fichier.wav>")
        sys.exit(1)

    predict_file(sys.argv[1])
