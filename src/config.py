from pathlib import Path

# Répertoire racine du projet (= dossier contenant README.md)
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossiers de données
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Dossier pour les figures (spectrogrammes)
FIGURES_DIR = BASE_DIR / "assets" / "figures"

# Paramètres audio généraux
TARGET_SR = 22050  # Hz, fréquence d'échantillonnage de travail

# Filtre passe-bande FIR (en Hz)
BANDPASS_LOW = 400.0
BANDPASS_HIGH = 8000.0
FIR_NUMTAPS = 1025  # ordre du filtre + 1 (doit être impair de préférence)

# Paramètres STFT
STFT_N_FFT = 1024
STFT_HOP_LENGTH = 256
STFT_WINDOW = "hann"

# Création des dossiers si besoin
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
