![verdier-deurope](https://github.com/user-attachments/assets/ed3c5e9e-2d56-4fd4-8914-c0d80e39a733)
#  Biodiversity Monitoring Project  
Analyse automatique de chants d’oiseaux — Filtrage FIR, STFT & Spectrogrammes



##  Objectif

Ce projet permet de :

- charger automatiquement des fichiers `.wav` d’oiseaux,
- appliquer un **filtre passe-bande FIR** (400 Hz – 8 kHz),
- calculer un **spectrogramme STFT**,
- sauvegarder les **audios filtrés** et les **figures** générées.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/DiegoRadigues/Biodiversity-Monitoring-Project.git
cd Biodiversity-Monitoring-Project
```

### 2. Créer un environnement virtuel

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installer les dépendances

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

##  Lancer le pipeline

```powershell
python scripts/run_pipeline.py
```

##  Structure du projet

```
Biodiversity-Monitoring-Project/
├── data/
│   ├── raw/
│   └── processed/
├── assets/
│   └── figures/
├── scripts/
│   └── run_pipeline.py
└── src/
    ├── config.py
    ├── io/
    ├── signal/
    └── detection/
```

##  Auteurs

- Diego de Radigues  
- Arthur Dufour  
- Ange Simpalingabo  
