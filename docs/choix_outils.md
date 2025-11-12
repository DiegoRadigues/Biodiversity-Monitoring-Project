# Choix des outils & requirements 
**Arthur DUFOUR — Ange SIMPALINGABO — Diego de RADIGUES**



## 1) Outils retenus (et pourquoi)

### Python (choix principal)
- **Pourquoi** : bon écosystème audio/signal, facile à automatiser avec scripts. reproductibilité.
- **Utilisé pour** : lecture/écriture `.wav`, filtrage FIR, STFT/mel‑spectrogramme, détection par flux spectral + seuil adaptatif, features (MFCC, centroid, roll‑off, entropie), métriques et graphiques.

### Support
- **Audacity** : écoute rapide, inspection visuelle. utile pour vérifier le filtrage. déjà utilisé et simple à maitriser


## 2) Requirements Python (évoluera peut-être au fil du projet, à update)

Fichier : **[../../requirements.txt](./requirements.txt)**

- `numpy`, `scipy` → calculs numériques, filtrage FIR, STFT.  
- `librosa`, `soundfile` → I/O audio, MFCC, aides spectrogramme.  
- `matplotlib` → figures (spectrogrammes, réponses en fréquence, overlays).  
- `pandas` → tables `segments.csv` / `features.csv`.  
- `scikit-learn` → stats de base et petites classifications si besoin.  
- `tqdm`, `joblib`, `numba` → confort/performances (barres de progression, parallélisation légère, accélération de certaines fonctions).

> Pas de deep learning mais sinon ça reste possible en python


---

## 3) Installation (Windows PowerShell)

> À exécuter à la racine du projet : `PS C:\Users\diego\Biodiversity-Monitoring-Project>`

```powershell
# 1) Créer un environnement virtuel (évite les conflits système)
python -m venv .venv

# 2) Activer l'environnement
.venv\Scripts\activate

# 3) Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```


