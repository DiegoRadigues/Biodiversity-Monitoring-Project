from pathlib import Path
import soundfile as sf
import numpy as np


def save_audio(path: Path, y: np.ndarray, sr: int) -> None:
    """
    Sauvegarde un signal mono au format .wav (float32).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(path, y.astype("float32"), sr)
