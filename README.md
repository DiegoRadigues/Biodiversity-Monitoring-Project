![verdier-deurope](https://github.com/user-attachments/assets/ed3c5e9e-2d56-4fd4-8914-c0d80e39a733)
# Biodiversity Monitoring Project  
Analyse automatique de chants d’oiseaux — Filtrage FIR, STFT, spectrogrammes et classification par segments

---

## 1. Contexte et objectifs du projet

Ce projet a été réalisé dans le cadre d’un travail d’analyse de signaux audio appliqué aux chants d’oiseaux.

L’objectif principal est de construire un **pipeline complet, reproductible et documenté** permettant de :

1. Charger automatiquement des enregistrements `.wav` d’oiseaux déposés dans `data/raw/`.
2. Appliquer un **filtre passe-bande FIR à phase linéaire** (bande de référence 400 Hz – 8 kHz).
3. Calculer un **spectrogramme STFT** et l’enregistrer sous forme d’images.
4. Détecter automatiquement les segments contenant de l’activité acoustique (chant).
5. Sauvegarder ces segments dans un fichier `segments.csv`.
6. Extraire des **caractéristiques audio (MFCC, descripteurs spectraux, F0)** pour chaque segment.
7. Entraîner un **classifieur par segments (Random Forest)** afin de distinguer différentes espèces.
8. Utiliser ce modèle pour prédire l’espèce dominante d’un **nouvel enregistrement `.wav`**.

L’idée est que, à partir de fichiers audio bruts d’oiseaux, on arrive à :

- des **figures** (spectrogrammes simples et annotés),
- des **tables** (`segments.csv`, `features.csv`, `predictions_segments.csv`),
- un **modèle appris** (`rf_segments.joblib`),
- et un **script de prédiction** pour de nouveaux fichiers.

Ce README sert de **rapport autonome** : il résume la méthode, la structure du dépôt, les choix techniques et présente des exemples de résultats.

---

## 2. Installation et environnement

### 2.1. Cloner le dépôt

```bash
git clone https://github.com/DiegoRadigues/Biodiversity-Monitoring-Project.git
cd Biodiversity-Monitoring-Project
```

### 2.2. Créer un environnement virtuel (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2.3. Installer les dépendances Python

Les dépendances principales (détails dans `docs/choix_outils.md` et `requirements.txt`) sont :

- `numpy`, `scipy` : calculs numériques, filtrage FIR, STFT
- `librosa`, `soundfile` : I/O audio, MFCC, spectrogrammes
- `matplotlib` : figures, spectrogrammes, matrice de confusion
- `pandas` : tables (`segments.csv`, `features.csv`, `predictions_segments.csv`)
- `scikit-learn` : classification, métriques
- `tqdm`, `joblib`, `numba` : confort et performances (facultatif)

Installation :

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> Version recommandée de Python : 3.10 ou plus récent.

---

## 3. Organisation du dépôt

Le projet est structuré pour séparer clairement données, code réutilisable, scripts d’entrée et résultats d’expériences.

### 3.1. Vue d’ensemble

```text
Biodiversity-Monitoring-Project/
├── README.md                          # Ce document
├── requirements.txt
├── docs/
│   ├── analyse_filtrage.md           # Choix du FIR, paramètres STFT
│   ├── choix_outils.md               # Justification des outils Python
│   └── Analyses_des_chants_d_oiseaux_presentation.pdf
├── project_plan/
│   ├── ARBORESCENCE.md               # Spécification de l’architecture
│   └── WBS.md                        # Work Breakdown Structure détaillé
├── data/
│   ├── raw/                          # Fichiers .wav d’origine
│   │   └── Bird songs/
│   │       ├── grive.wav
│   │       ├── merle.wav
│   │       ├── mésange charbonnière.wav
│   │       ├── pic vert.wav
│   │       ├── pie.wav
│   │       ├── pigeon.wav
│   │       ├── pinson.wav
│   │       ├── rouge-gorge.wav
│   │       ├── sitelle torchepot.wav
│   │       └── tourterelle.wav
│   └── processed/                    # Audios filtrés (FIR) générés par le pipeline
│       └── Bird songs/
│           ├── grive_bandpass.wav
│           ├── merle_bandpass.wav
│           ├── ...
├── assets/
│   ├── figures/                      # Spectrogrammes simples + matrice de confusion
│   │   ├── Bird songs/
│   │   │   ├── grive_spectrogram.png
│   │   │   ├── merle_spectrogram.png
│   │   │   └── ...
│   │   └── rf_segments_confusion_matrix.png
│   └── figures_annotated/            # Spectrogrammes annotés (segments)
│       ├── grive_segments.png
│       ├── merle_segments.png
│       └── ...
├── src/                              # Code réutilisable (package)
│   ├── __init__.py
│   ├── config.py                     # Paramètres centraux (chemins, FIR, STFT, SR)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── load_audio.py             # Lecture, normalisation, resampling
│   │   └── save_audio.py             # Sauvegarde wav
│   ├── signal/
│   │   ├── __init__.py
│   │   ├── fir_bandpass.py           # Conception + application du FIR linéaire
│   │   ├── stft.py                   # STFT + sauvegarde de spectrogrammes
│   │   └── noise_reduction.py        # Réservé pour des améliorations futures
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── vad_spectral_flux.py      # Détection d’activité par flux spectral
│   │   └── segmentation_2d.py        # Squelette pour segmentation 2D (non utilisé)
│   └── features/
│       ├── __init__.py
│       ├── mfcc.py                   # MFCC + statistiques (mean, std, min, max)
│       ├── spectral_stats.py         # Centroid, bandwidth, rolloff, flatness, entropie
│       └── f0.py                     # Estimation de F0 (YIN) + statistiques
├── scripts/                          # Scripts d’entrée (cliquables)
│   ├── run_pipeline.py               # Pipeline complet jusqu’aux segments
│   ├── extract_features.py           # Extraction des features par segment
│   ├── train_classifier.py           # Entraînement RandomForest + rapport + figures
│   ├── predict_segments.py           # Prédiction de chaque segment du dataset
│   ├── inspect_segments.py           # Génération des spectrogrammes annotés
│   └── predict_file.py               # Prédiction de l’espèce dominante d’un nouveau .wav
└── experiments/
    └── exp1/
        ├── segments.csv              # Segments détectés (fichier, t_onset, t_offset)
        ├── features.csv              # Features agrégées par segment
        ├── metrics.json              # (optionnel) métriques de détection
        ├── predictions_segments.csv  # Prédictions du modèle sur les segments
        ├── models/
        │   └── rf_segments.joblib    # Modèle RandomForest sauvegardé
        └── reports/
            └── rf_segments_report.txt# Rapport de classification (sklearn)
```

Pour des détails supplémentaires, se référer à `project_plan/ARBORESCENCE.md` qui décrit le contenu minimum par dossier.

---

## 4. Méthodes et choix de traitement du signal

Les choix méthodologiques et les justifications théoriques sont détaillés dans :

- `docs/analyse_filtrage.md`
- `docs/choix_outils.md`

Cette section résume les points principaux.

### 4.1. Filtrage passe-bande FIR (0,4 – 8 kHz)

Les chants d’oiseaux étudiés sont principalement situés entre **1 kHz et 8 kHz**. Afin de :

- rejeter les bruits graves (vent, trafic, manipulations de micro, etc.),
- ne pas conserver des fréquences très aiguës généralement peu informatives,

un **filtre passe-bande FIR à phase linéaire** est utilisé, avec :

- coupure basse : `BANDPASS_LOW = 400.0` Hz  
- coupure haute : `BANDPASS_HIGH = 8000.0` Hz  
- nombre de taps : `FIR_NUMTAPS = 1025`

Implémentation : `src/signal/fir_bandpass.py`

```python
from src.signal.fir_bandpass import apply_fir_bandpass

y_filt = apply_fir_bandpass(
    y,
    fs=sr,
    lowcut=BANDPASS_LOW,
    highcut=BANDPASS_HIGH,
    numtaps=FIR_NUMTAPS,
)
```

Le filtrage est appliqué hors temps réel, via `scipy.signal.filtfilt`, ce qui garantit une **phase globale nulle** (pas de décalage temporel) et conserve l’enveloppe des cris.

### 4.2. STFT et spectrogrammes

Les paramètres STFT sont centralisés dans `src/config.py` :

- `TARGET_SR = 22050` Hz
- `STFT_N_FFT = 1024`
- `STFT_HOP_LENGTH = 256`
- `STFT_WINDOW = "hann"`

Calcul et sauvegarde d’un spectrogramme typique (dans `run_pipeline.py`) :

```python
from src.signal.stft import compute_spectrogram, save_spectrogram_figure

S_db, freqs, times = compute_spectrogram(
    y_filt,
    sr=sr,
    n_fft=STFT_N_FFT,
    hop_length=STFT_HOP_LENGTH,
    window=STFT_WINDOW,
)

save_spectrogram_figure(
    S_db,
    sr=sr,
    hop_length=STFT_HOP_LENGTH,
    out_path=fig_out_path,
)
```

Les figures générées sont stockées dans `assets/figures/Bird songs/*_spectrogram.png`.

### 4.3. Détection d’activité par flux spectral

La détection d’activité (VAD “aviaire”) repose sur le **flux spectral** :

1. Calcul de la magnitude STFT.
2. Différence entre frames successives.
3. On ne garde que les augmentations (partie positive).
4. Seuil adaptatif basé sur la médiane et l’écart-type : `seuil = médiane + k·σ`.
5. Conversion en segments temporels et fusion des segments proches.

Implémentation principale : `src/detection/vad_spectral_flux.py`

Fonction utilisée dans le pipeline de base :

```python
from src.detection.vad_spectral_flux import detect_activity

segments = detect_activity(
    y_filt,
    sr=sr,
    n_fft=STFT_N_FFT,
    hop_length=STFT_HOP_LENGTH,
    k=0.6,
    min_duration_s=0.03,
)
```

Chaque segment est ensuite converti en intervalle temporel `(t_onset_s, t_offset_s)` et stocké dans `experiments/exp1/segments.csv`.

### 4.4. Extraction de features par segment

Les features sont calculées pour chaque segment audio détecté (voir `scripts/extract_features.py`).  
Lors de l’extraction, on charge d’abord l’audio filtré complet (`*_bandpass.wav`), puis on découpe selon les timestamps des segments.

Pour chaque segment, on calcule :

1. **MFCC** (`src/features/mfcc.py`)  
   - 20 coefficients MFCC
   - Agrégation : moyenne, écart-type, min, max pour chaque coefficient  
   → environ 80 colonnes au total pour les MFCC

2. **Descripteurs spectraux** (`src/features/spectral_stats.py`)  
   - centroid spectral
   - bande passante
   - rolloff (95 %)
   - flatness
   - entropie spectrale moyenne  
   → moyenne et écart-type pour certains, soit plusieurs colonnes additionnelles

3. **Fréquence fondamentale F0** (`src/features/f0.py`)  
   - estimation par l’algorithme YIN (`librosa.yin`)
   - `f0_mean_hz`, `f0_std_hz`, `f0_min_hz`, `f0_max_hz`
   - `f0_voiced_ratio` (proportion de frames voisées)

Extrait du code d’agrégation (simplifié) :

```python
from src.features.mfcc import compute_mfcc_stats
from src.features.spectral_stats import compute_spectral_stats
from src.features.f0 import compute_f0_stats

def extract_features_for_segment(y, sr):
    features = {}
    features.update(compute_mfcc_stats(y, sr, n_mfcc=20,
                                       n_fft=STFT_N_FFT,
                                       hop_length=STFT_HOP_LENGTH))
    features.update(compute_spectral_stats(y, sr,
                                           n_fft=STFT_N_FFT,
                                           hop_length=STFT_HOP_LENGTH))
    features.update(compute_f0_stats(y, sr,
                                     fmin=300.0,
                                     fmax=8000.0,
                                     frame_length=STFT_N_FFT,
                                     hop_length=STFT_HOP_LENGTH))
    return features
```

Les résultats sont sauvegardés dans `experiments/exp1/features.csv` avec au minimum les colonnes :

- `file`, `segment_id`, `t_onset_s`, `t_offset_s`, `duration_s`
- `mfcc_01_mean`, `mfcc_01_std`, ..., `mfcc_20_max`
- `spec_centroid_hz_mean`, `spec_centroid_hz_std`, etc.
- `f0_mean_hz`, `f0_std_hz`, `f0_voiced_ratio`, etc.

### 4.5. Classification par segments (RandomForest)

Un classifieur standard de type **RandomForest** est entraîné à partir des features.

Script : `scripts/train_classifier.py`

Pipeline utilisé :

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )),
    ]
)
```

1. Construction de la colonne `species` à partir du nom de fichier (ex. `Bird songs/grive.wav` → `grive`).  
2. Séparation en train / test (25 % test, stratifié par espèce).  
3. Entraînement du modèle.  
4. Calcul d’un **rapport de classification** (F1, précision, rappel par classe).  
5. Calcul et sauvegarde d’une **matrice de confusion** dans `assets/figures/rf_segments_confusion_matrix.png`.  
6. Sauvegarde du modèle dans `experiments/exp1/models/rf_segments.joblib` avec la liste des colonnes de features (`feature_cols`).

Le rapport texte complet est enregistré dans :  
`experiments/exp1/reports/rf_segments_report.txt`.

---

## 5. Exécution du pipeline : scripts et exemples

### 5.1. Étape 1 — Prétraitement et détection de segments

Commandes :

```powershell
# À la racine du projet
python scripts/run_pipeline.py
```

Effets principaux :

- lecture des fichiers `.wav` dans `data/raw/Bird songs/`
- filtrage passe-bande FIR
- sauvegarde des audios filtrés dans `data/processed/Bird songs/*_bandpass.wav`
- calcul et sauvegarde des spectrogrammes dans `assets/figures/Bird songs/*_spectrogram.png`
- détection de segments et création de `experiments/exp1/segments.csv`

Exemple de contenu de `segments.csv` (extrait simplifié) :

```csv
file,segment_id,t_onset_s,t_offset_s
Bird songs/merle.wav,0,0.35,0.72
Bird songs/merle.wav,1,0.90,1.25
Bird songs/grive.wav,0,0.40,0.78
...
```

### 5.2. Étape 2 — Extraction de features par segment

Commandes :

```powershell
python scripts/extract_features.py
```

Ce script :

- lit `experiments/exp1/segments.csv`,
- charge les audios filtrés correspondants (`*_bandpass.wav`),
- découpe les segments, calcule les features,
- sauvegarde le résultat dans `experiments/exp1/features.csv`.

Exemple (structure partielle de `features.csv`) :

```csv
file,segment_id,t_onset_s,t_offset_s,duration_s,mfcc_01_mean,mfcc_01_std,...,f0_mean_hz,f0_voiced_ratio
Bird songs/merle.wav,0,0.35,0.72,0.37,-245.1,12.3,...,3200.5,0.92
Bird songs/merle.wav,1,0.90,1.25,0.35,-240.8,11.7,...,3150.7,0.89
...
```

### 5.3. Étape 3 — Entraînement du classifieur

Commandes :

```powershell
python scripts/train_classifier.py
```

Ce script :

- lit `experiments/exp1/features.csv`,
- crée la colonne `species`,
- sépare train / test,
- entraîne un RandomForest,
- génère :
  - `experiments/exp1/reports/rf_segments_report.txt`
  - `assets/figures/rf_segments_confusion_matrix.png`
  - `experiments/exp1/models/rf_segments.joblib`

Extrait typique de la sortie console (simplifié) :

```text
Chargement des features depuis : experiments/exp1/features.csv

Nombre de segments par espèce :
merle              45
grive              42
mésange charbonnière 39
...

Nombre de features : 120
Nombre total d'échantillons : 200

Taille train : 150
Taille test  : 50

Entraînement du modèle...

=== Rapport de classification (test) ===
              precision    recall  f1-score   support

       grive       0.90      0.90      0.90        10
       merle       0.89      0.89      0.89         9
        ...       ...       ...       ...       ...

accuracy                           0.90        50
```

La matrice de confusion finale est sauvegardée sous forme d’image :  
`assets/figures/rf_segments_confusion_matrix.png`.

### 5.4. Étape 4 — Prédiction des segments du dataset

Une fois le modèle entraîné, on peut prédire l’espèce pour tous les segments de `features.csv` :

```powershell
python scripts/predict_segments.py
```

Ce script :

- charge `features.csv` ainsi que le modèle `rf_segments.joblib`,
- applique le classifieur sur chaque ligne,
- calcule une précision globale sur l’ensemble du dataset,
- enregistre les prédictions dans `experiments/exp1/predictions_segments.csv`.

Extrait de sortie console :

```text
Accuracy sur l'ensemble des segments : 0.920

Quelques lignes :
                     file  segment_id  t_onset_s  t_offset_s  species  pred_species  pred_confidence
0      Bird songs/merle.wav           0      0.350       0.720    merle        merle          0.98
1      Bird songs/merle.wav           1      0.900       1.250    merle        merle          0.95
...
```

### 5.5. Étape 5 — Visualisation des segments annotés

Pour mieux comprendre la détection, on peut générer des spectrogrammes avec les segments surlignés :

```powershell
python scripts/inspect_segments.py
```

Ce script produit des images dans `assets/figures_annotated/`, par exemple :

- `assets/figures_annotated/merle_segments.png`
- `assets/figures_annotated/grive_segments.png`

Ces figures montrent les zones détectées (segments) en superposition sur le spectrogramme.

### 5.6. Étape 6 — Prédire l’espèce dominante d’un nouveau fichier `.wav`

Enfin, on peut appliquer le pipeline à un nouveau fichier (hors dataset de base) :

```powershell
python scripts/predict_file.py pathers
ouvel_enregistrement.wav
```

Le script effectue :

1. chargement et resampling du fichier audio,
2. filtrage FIR passe-bande,
3. détection des segments par VAD,
4. extraction des features pour chaque segment,
5. prédiction de l’espèce pour chaque segment,
6. vote majoritaire pour déterminer l’espèce dominante.

Exemple de sortie (simplifiée) :

```text
=== Analyse de nouveau_fichier.wav ===
Audio chargé : durée = 12.34 s, sr = 22050 Hz
8 segments détectés
  - segment 0: 0.32s → 0.71s → merle (conf=0.96)
  - segment 1: 0.90s → 1.20s → merle (conf=0.93)
  ...
=== Espèce dominante prédite ===
merle
```

---

## 6. Exemples de figures

Les figures les plus importantes générées par le projet sont :

- Spectrogrammes de base (filtrés) pour chaque espèce, par exemple :
  - `assets/figures/Bird songs/merle_spectrogram.png`
  - `assets/figures/Bird songs/grive_spectrogram.png`
  - etc.

- Spectrogrammes annotés par segments :
  - `assets/figures_annotated/merle_segments.png`
  - `assets/figures_annotated/grive_segments.png`
  - ...

- Matrice de confusion du classifieur par segments :
  - `assets/figures/rf_segments_confusion_matrix.png`

Ces figures sont mobilisées dans le fichier `Analyses_des_chants_d_oiseaux_presentation.pdf` pour illustrer les résultats.

---

## 7. Limites, pistes d’amélioration et perspectives

Ce projet constitue un **MVP** (Minimum Viable Product) pour la détection et la classification de chants d’oiseaux. Plusieurs limitations et pistes d’amélioration sont identifiées dans `project_plan/WBS.md` et résumées ici :

- **Nombre d’espèces limité** : le dataset travaille sur un petit ensemble d’espèces (grive, merle, mésange, etc.).  
  → Extension possible vers plus d’espèces et plus d’enregistrements.

- **Pas de segmentation 2D complète** : le module `segmentation_2d.py` est présent mais non implémenté.  
  → Amélioration envisageable avec des techniques de segmentation sur le plan temps–fréquence (masques binaires, composantes connexes, etc.).

- **Réduction de bruit optionnelle non utilisée** : `noise_reduction.py` est prévu pour une éventuelle soustraction spectrale ou filtre médian 2D, mais n’est pas activement utilisé dans la version actuelle.  
  → Intégrer une réduction de bruit contrôlée pour améliorer la robustesse en conditions réelles (vent, pluie, insectes).

- **Modèle de classification simple** : le RandomForest sur features classiques fonctionne correctement, mais :  
  → des modèles plus avancés (SVM, gradient boosting, CNN avec spectrogrammes log-Mel) pourraient être explorés si l’on élargit le jeu de données.

- **Pas de temps réel** : tout le pipeline est hors ligne, comme prévu dans les hypothèses de départ.  
  → Une version temps réel nécessiterait un filtrage IIR et une gestion des buffers en streaming.

Malgré ces limites, le projet met en place une base solide pour des travaux futures en bioacoustique : pipeline reproductible, code organisé, documentation et premiers résultats quantitatifs.

---

## 8. Références internes au dépôt

Pour aller plus loin, les fichiers internes suivants documentent le projet :

- `docs/analyse_filtrage.md` : justification des paramètres du filtre FIR et de la STFT.
- `docs/choix_outils.md` : choix techniques (Python, librairies) et installation.
- `project_plan/ARBORESCENCE.md` : spécification initiale de l’architecture du dépôt.
- `project_plan/WBS.md` : découpage du travail, jalons, checklist de validation.

Les résultats chiffrés détaillés sont disponibles dans :

- `experiments/exp1/reports/rf_segments_report.txt`
- `experiments/exp1/predictions_segments.csv`
- `experiments/exp1/metrics.json` (si renseigné)

---

## 9. Auteurs

Projet réalisé par :

- Diego de Radigues  
- Arthur Dufour  
- Ange Simpalingabo  

Sous la supervision de l’équipe pédagogique, dans le cadre d’un projet d’analyse de signaux audio appliqué à la biodiversité.
