import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib


# chemins de base
ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1"
FEATURES_PATH = EXP_DIR / "features.csv"

MODELS_DIR = EXP_DIR / "models"
REPORTS_DIR = EXP_DIR / "reports"
FIGURES_DIR = ROOT / "assets" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Chargement des features depuis : {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)

    # --- création de la colonne 'species' à partir du nom du fichier ---
    # "Bird songs\\grive.wav" -> "grive"
    df["species"] = df["file"].apply(lambda s: Path(s).stem)

    print("\nNombre de segments par espèce :")
    print(df["species"].value_counts())

    # --- définition des features X et de la cible y ---
    drop_cols = ["file", "segment_id", "t_onset_s", "t_offset_s", "species"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df["species"].values

    print(f"\nNombre de features : {X.shape[1]}")
    print(f"Nombre total d'échantillons : {X.shape[0]}")

    # --- train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print(f"\nTaille train : {X_train.shape[0]}")
    print(f"Taille test  : {X_test.shape[0]}")

    # --- pipeline : standardisation + RandomForest ---
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nEntraînement du modèle...")
    clf.fit(X_train, y_train)

    # --- évaluation ---
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("\n=== Rapport de classification (test) ===")
    print(report)

    # sauvegarde du rapport texte
    report_path = REPORTS_DIR / "rf_segments_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RandomForest sur segments (features MFCC + spectre + F0)\n\n")
        f.write(report)

    print(f"\nRapport sauvegardé -> {report_path}")

    # --- matrice de confusion ---
    labels = sorted(df["species"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Matrice de confusion (segments)")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_ylabel("Vérité terrain")
    ax.set_xlabel("Prédiction")

    # annotations sur la matrice
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    cm_path = FIGURES_DIR / "rf_segments_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"Matrice de confusion sauvegardée -> {cm_path}")

    # --- sauvegarde du modèle ---
    model_path = MODELS_DIR / "rf_segments.joblib"
    joblib.dump({"pipeline": clf, "feature_cols": feature_cols}, model_path)
    print(f"Modèle sauvegardé -> {model_path}")


if __name__ == "__main__":
    main()
