import sys
from pathlib import Path
import pandas as pd

# Ajouter la racine du projet dans sys.path pour trouver "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    FIGURES_DIR,
    TARGET_SR,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FIR_NUMTAPS,
    STFT_N_FFT,
    STFT_HOP_LENGTH,
    STFT_WINDOW,
)
from src.io.load_audio import load_audio
from src.io.save_audio import save_audio
from src.signal.fir_bandpass import apply_fir_bandpass
from src.signal.stft import compute_spectrogram, save_spectrogram_figure
from src.detection.vad_spectral_flux import detect_activity


# Dossier des résultats de l'expérience
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp1"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_CSV = EXPERIMENT_DIR / "segments.csv"


def process_file(wav_path: Path, all_segments: list) -> None:
    print(f"\n=== Traitement : {wav_path} ===")

    # 1) Charger l'audio
    y, sr = load_audio(wav_path, target_sr=TARGET_SR)
    print(f"  - signal chargé, sr = {sr} Hz, durée = {len(y) / sr:.2f} s")

    # 2) Filtrage passe-bande
    y_filt = apply_fir_bandpass(
        y,
        fs=sr,
        lowcut=BANDPASS_LOW,
        highcut=BANDPASS_HIGH,
        numtaps=FIR_NUMTAPS,
    )

    # 3) Chemin de sortie pour l'audio filtré
    rel_path = wav_path.relative_to(DATA_RAW)
    out_wav_path = DATA_PROCESSED / rel_path
    out_wav_path = out_wav_path.with_name(out_wav_path.stem + "_bandpass.wav")

    save_audio(out_wav_path, y_filt, sr)
    print(f"  - audio filtré sauvegardé -> {out_wav_path}")

    # 4) Spectrogramme
    S_db, freqs, times = compute_spectrogram(
        y_filt,
        sr=sr,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        window=STFT_WINDOW,
    )

    # 5) Sauvegarde de la figure
    fig_rel = rel_path.with_name(rel_path.stem + "_spectrogram.png")
    fig_out_path = FIGURES_DIR / fig_rel

    save_spectrogram_figure(
        S_db,
        sr=sr,
        hop_length=STFT_HOP_LENGTH,
        out_path=fig_out_path,
    )
    print(f"  - spectrogramme sauvegardé -> {fig_out_path}")

    # 6) Détection VAD (flux spectral)
    segments = detect_activity(
        y_filt,
        sr=sr,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        k=0.6,
        min_duration_s=0.03,
    )
    print(f"  - {len(segments)} segments détectés")

    # 7) Ajouter les segments à la liste globale
    rel_str = str(rel_path)
    for seg_id, (t_on, t_off) in enumerate(segments):
        all_segments.append({
            "file": rel_str,
            "segment_id": seg_id,
            "t_onset_s": t_on,
            "t_offset_s": t_off,
        })


def main():
    wav_files = list(DATA_RAW.rglob("*.wav"))

    if not wav_files:
        print(f"Aucun fichier .wav trouvé dans {DATA_RAW}")
        return

    print(f"{len(wav_files)} fichiers .wav trouvés.")

    all_segments = []

    for wav_path in wav_files:
        process_file(wav_path, all_segments)

    # Sauvegarder les segments
    if all_segments:
        df = pd.DataFrame(all_segments)
        df.to_csv(SEGMENTS_CSV, index=False)
        print(f"\nSegments sauvegardés dans : {SEGMENTS_CSV}")
    else:
        print("\nAucun segment détecté.")


if __name__ == "__main__":
    main()
