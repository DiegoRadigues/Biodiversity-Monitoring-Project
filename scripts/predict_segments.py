import pandas as pd
from pathlib import Path
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1"
FEATURES_PATH = EXP_DIR / "features.csv"
MODEL_PATH = EXP_DIR / "models" / "rf_segments.joblib"
OUT_PATH = EXP_DIR / "predictions_segments.csv"


def main():
    print(f"Chargement des features : {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)

    # même colonne species que dans train_classifier.py
    df["species"] = df["file"].apply(lambda s: Path(s).stem)

    print(f"Chargement du modèle : {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    X = df[feature_cols].values

    print("Prédiction des segments...")
    y_pred = pipeline.predict(X)

    # si le modèle supporte predict_proba, on récupère la confiance
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        max_proba = np.max(proba, axis=1)
    else:
        max_proba = np.nan * np.ones(len(df))

    df["pred_species"] = y_pred
    df["pred_confidence"] = max_proba

    # petit check de précision globale sur tout le dataset
    accuracy = (df["pred_species"] == df["species"]).mean()
    print(f"\nAccuracy sur l'ensemble des segments : {accuracy:.3f}")

    print("\nQuelques lignes :")
    print(
        df[
            [
                "file",
                "segment_id",
                "t_onset_s",
                "t_offset_s",
                "species",
                "pred_species",
                "pred_confidence",
            ]
        ].head(15)
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"\nPrédictions sauvegardées -> {OUT_PATH}")


if __name__ == "__main__":
    main()
