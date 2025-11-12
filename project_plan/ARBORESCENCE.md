# ARBORESCENCE — Biodiversity Monitoring Project 

> **But** : partir d’un fichier .wav téléchargé, filtrer (FIR linéaire), calculer STFT, détecter/segmenter, extraire des paramètres (MFCC, stats spectrales) et stocker les résultats et figures.  
> **Principe** : ne garder que ce qui sera effectivement utilisé pour un pipeline “hors temps réel”.

```
Biodiversity-Monitoring-Project/
├─ README.md                          # Guide rapide d’exécution (installation + commande de pipeline)
├─ ARBORESCENCE.md                    # Ce document
├─ docs/                              # Documents fournis / rapports courts
│  ├─ analyse_filtrage.md             # Notes sur le choix FIR + paramètres
│  └─ Analyses_des_chants_d_oiseaux_presentation.pdf
├─ project_plan/
│  └─ WBS.md                          # Décomposition du travail validée
├─ data/
│  ├─ raw/                            # Fichiers .wav d’origine (source téléchargée)
│  └─ processed/                      # Audios filtrés / resamplés (.wav) et artefacts intermédiaires
├─ assets/
│  └─ figures/                        # Spectrogrammes, réponses en fréquence, visuels des segments
├─ src/                               # Code réutilisable (organisé par domaine)
│  ├─ __init__.py
│  ├─ config.py                       # Paramètres centraux : fs, bandes du FIR, STFT, chemins par défaut
│  ├─ io/
│  │  ├─ __init__.py
│  │  ├─ load_audio.py                # Lecture mono + resample + normalisation
│  │  └─ save_audio.py                # Écriture .wav intermédiaires (processed/)
│  ├─ signal/
│  │  ├─ __init__.py
│  │  ├─ fir_bandpass.py              # Conception + application du filtre FIR (phase linéaire)
│  │  ├─ stft.py                      # Calcul STFT + helpers d’affichage
│  │  └─ noise_reduction.py           # (Optionnel) soustraction spectrale / filtre médian 2D
│  ├─ detection/
│  │  ├─ __init__.py
│  │  ├─ vad_spectral_flux.py         # Détection (RMS/flux spectral) + seuillage adaptatif
│  │  └─ segmentation_2d.py           # Masque binaire sur spectrogramme + composantes connexes
│  └─ features/
│     ├─ __init__.py
│     ├─ mfcc.py                      # MFCC (+Δ/ΔΔ)
│     ├─ spectral_stats.py            # Centroid, roll-off, entropie, flatness, bande passante
│     └─ f0.py                        # Estimation f0 (autocorrélation/cepstre) avec bornes
├─ scripts/                           # Points d’entrée “prêts à lancer”
│  ├─ run_pipeline.py                 # End-to-end : data/raw/*.wav → processed/, segments.csv, figures/
│  └─ extract_features.py             # À partir de segments → features.csv/parquet (+ stats)
└─ experiments/
   └─ exp1/                           # Résultats d’exécution (un dossier par expérience)
      ├─ segments.csv                 # Onset/offset, fmin/fmax, score
      ├─ features.csv                 # (généré par extract_features.py)
      ├─ metrics.json                 # Métriques de détection (F1, IoU)
      └─ figures/                     # Figures générées (spectrogrammes, overlays)
```

---

## Détails de contenu (minimum viable)

- **data/raw/** : placez ici les .wav **téléchargés** (sources originales, non modifiées).
- **data/processed/** : .wav **filtrés** (FIR 0,4–8 kHz par défaut), plus tout audio intermédiaire utile.
- **assets/figures/** : export des **spectrogrammes** (STFT), réponses en fréquence du FIR, et visuels de **segments**.
- **src/config.py** : un seul endroit pour régler les **paramètres** (fs, fréquences de coupure, taille fenêtre/hop STFT, chemins par défaut).
- **src/io/** : fonctions d’entrée/sortie audio (lecture/écriture), pour garder les scripts simples.
- **src/signal/** : traitements signal “purs” (filtrage FIR, STFT, débruitage optionnel).
- **src/detection/** : **détection** d’activité + **segmentation** 2D des régions temps–fréquence.
- **src/features/** : extraction des **MFCC**, statistiques spectrales et **f0**.
- **scripts/run_pipeline.py** : pipeline unique : _raw → processed → détection/segmentation → figures + segments.csv_.
- **scripts/extract_features.py** : lit `segments.csv`, **extrait** les features, sauvegarde `features.csv`.
- **experiments/exp1/** : premier run reproductible avec **résultats** et **figures**.

> Pas de dossiers “notebooks”/“models”/“reports” ici pour rester minimal : ajoutez-les **uniquement** si vous en avez besoin.

---

