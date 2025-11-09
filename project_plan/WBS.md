# WBS — Projet d'analyse de chants d'oiseaux 

> **Objectif** : partir d’un enregistrement .wav téléchargé, détecter les cris, extraire des paramètres pertinents et préparer une base pour une identification d’espèce.  

> **Hypothèse de départ** : traitement **hors temps réel** sur poste, audio mono, 22,05–44,1 kHz, 16 bits.

---

## 1. Pilotage & cadrage
### 1.1. Portée & exigences
- 1.1.1. Définir les livrables (pipeline, rapport, figures, tableaux de paramètres, baseline de classification)
- 1.1.2. Fixer les métriques d’évaluation (détection : F1, IoU±50 ms ; classification : F1 macro, confusion)
- 1.1.3. Définir les espèces cibles (2–5 pour un POC) et les scénarios (oiseaux dominants, bruit météo)
- 1.1.4. Fixer l’environnement (Python ≥ 3.10, packages : numpy, scipy, librosa, scikit‑learn, matplotlib)
- 1.1.5. Contraintes : pas de temps réel, fichiers .wav, phase linéaire souhaitée

### 1.2. Organisation & rôles
non définis pour le moment

### 1.3. Planning & jalons
- 1.3.1. J0–J7 : Baseline détection + spectrogrammes
- 1.3.2. J8–J14 : Extraction de paramètres + tableaux
- 1.3.3. J15–J21 : Baseline classification (optionnelle)
- 1.3.4. J22–J28 : Consolidation, comparaisons, rapport final

---

## 2. Données & conformité
### 2.1. Sourcing & licence
- 2.1.1. Importer l’audio téléchargé (.wav non compressé recommandé)
- 2.1.2. Vérifier droits d’usage/citation (si open data : noter la licence)
- 2.1.3. Renseigner métadonnées de base (source, date, lieu, matériel si connu)

### 2.2. Structure de répertoires
- 2.2.1. `data/raw/` (originaux), `data/processed/` (signaux filtrés), `assets/` (figures)
- 2.2.2. `experiments/` (configs, résultats, logs), `reports/` (notebooks, PDF) `docs/` (documentation technique)

### 2.3. Traçabilité
- 2.3.1. Fichier `DATASET.md` (sources, versions, paramètres d’import)
- 2.3.2. Hash/checksum des fichiers bruts (optionnel)

---

## 3. Pré‑traitements
### 3.1. Lecture & normalisation
- 3.1.1. Charger en mono ; resampler si nécessaire (22,05 kHz conseillé)
- 3.1.2. Normaliser RMS/peak (documentation du gain appliqué)

### 3.2. Filtrage passe‑bande
- 3.2.1. Choisir FIR linéaire‑phase (fenêtre Hamming/Blackman, ordre adapté)
- 3.2.2. Bande de référence : 0,4–8 kHz (ajustable selon espèce/spectrogramme)
- 3.2.3. Valider réponses impulsionnelle et fréquentielle (courbes de Bode, délai de groupe)
- 3.2.4. Export intermédiaire `data/processed/xxx_bandpass.wav`

### 3.3. Réduction de bruit (optionnel)
- 3.3.1. Profil de bruit (silence estimé) + soustraction spectrale OU filtre médian 2D sur spectrogramme
- 3.3.2. Contrôle des artefacts (musical noise) et comparaison AB

---

## 4. Représentations temps‑fréquence
### 4.1. Paramètres STFT
- 4.1.1. Fenêtre 20–40 ms, hop 10 ms (recouvrement 50–75 %), FFT 1024–2048
- 4.1.2. Fenêtre Hann/Hamming, amplitude/log‑puissance
- 4.1.3. Sauvegarder spectrogrammes (.png) pour inspection

### 4.2. Représentations dérivées
- 4.2.1. Mel‑spectrogramme (64–128 bandes) pour ML
- 4.2.2. CQT si besoin de haute résolution fréquentielle locale
- 4.2.3. Enveloppes temporelles (RMS, flux spectral)

---

## 5. Détection & segmentation
### 5.1. Détection d’activité acoustique (VAD aviaire)
- 5.1.1. Seuil adaptatif sur RMS/flux spectral (médiane glissante + k·σ)
- 5.1.2. Morphologie (dilatation/érosion) pour regrouper micro‑pauses
- 5.1.3. Filtre d’exclusion < ~800 Hz si espèces concernées > 1 kHz

### 5.2. Segmentation 2D
- 5.2.1. Seuillage du spectrogramme (Otsu/percentile) → masque binaire
- 5.2.2. Composantes connexes → boîtes temps‑fréquence (onset, offset, fmin, fmax)
- 5.2.3. Non‑Max Suppression pour doublons/chevauchements

### 5.3. Export des segments
- 5.3.1. Table `segments.csv` (t_onset, t_offset, f_min, f_max, score)
- 5.3.2. Découpes audio par segment (optionnel)

---

## 6. Extraction de paramètres (features)
### 6.1. Temps & rythme
- 6.1.1. Durée, intervalles, taux de répétition
- 6.1.2. Profil d’enveloppe (attaque, maintien, décroissance)

### 6.2. Spectral & harmonique
- 6.2.1. Centroid, bande passante, roll‑off (85/95 %), flatness, entropie
- 6.2.2. f0 (autocorrélation/cepstre) + contour ; présence d’harmoniques

### 6.3. MFCC & dérivées
- 6.3.1. 13–20 MFCC + Δ/ΔΔ
- 6.3.2. Statistiques par segment (moyenne, écart‑type, min, max, percentiles)

### 6.4. Export
- 6.4.1. `features.parquet/csv` (clé : id_segment)
- 6.4.2. Dictionnaire des features (`FEATURES.md`)

---

## 7. Classification (optionnelle si on a le temps)
### 7.1. Baseline “classique”
- 7.1.1. Jeu réduit, MFCC+stats → SVM/Random Forest
- 7.1.2. Validation croisée, F1 macro, confusion

### 7.2. Approche “deep” (normalement c'est souvent fait avec CNN mais ici hors du cadre)
- 7.2.1. Entrée : log‑Mel 128×T → CNN/CRNN
- 7.2.2. Transfert d’apprentissage (geler couches, fine‑tuning tête)
- 7.2.3. Augmentations : time‑stretch, pitch‑shift, bruit, SpecAugment

### 7.3. Sorties & seuils (idem)
- 7.3.1. Probabilités par espèce (multi‑label si besoin)
- 7.3.2. Calibration & choix des seuils par espèce

---

## 8. Évaluation & qualité
### 8.1. Protocoles
- 8.1.1. Split par enregistrement/site (éviter fuite de données)
- 8.1.2. Détection : précision, rappel, F1, IoU (tolérance 50 ms)
- 8.1.3. Classification : F1 macro, confusion, PR par classe

### 8.2. Analyses
- 8.2.1. Sensibilité aux paramètres STFT et au filtrage
- 8.2.2. Robustesse bruit (vent/pluie/insectes)
- 8.2.3. Erreurs typiques (chevauchements, fausses alarmes)

### 8.3. Qualité & revues
- 8.3.1. Revues pair‑à‑pair du code et des figures
- 8.3.2. Reproductibilité (seed, versions, configs sauvegardées)

---

## 9. Livrables & documentation
### 9.1. Code & artefacts
- 9.1.1. Scripts/notebooks (prétraitement, détection, features, modèle)
- 9.1.2. Figures (spectrogrammes annotés, distributions de features)
- 9.1.3. Tables (segments.csv, features.csv/parquet, métriques)

### 9.2. Rapport
- 9.2.1. Méthodes (justification des choix : FIR, STFT, paramètres)
- 9.2.2. Résultats (métriques, ablations, exemples audio/visuels)
- 9.2.3. Limites & perspectives (généralisation, temps réel, déploiement)

### 9.3. Guide d’exécution
- 9.3.1. README (installation, commande “end‑to‑end”)

---

## 10. Risques & parades
- 10.1. Bruit fort / faible SNR → renforcer filtrage, ajuster seuils, augmenter fenêtre
- 10.2. Déséquilibre d’espèces → pondération, échantillonnage, métriques macro
- 10.3. Chevauchement de cris → NMS, segmentation 2D, modèles séquentiels
- 10.4. Sur‑apprentissage → augmentation, validation stricte, early stopping
- 10.5. Données insuffisantes → open data d’appoint, transfert d’apprentissage

---

## 11. Checklists 
### 11.1. Pré‑traitement
- [ ] Mono + resample ok
- [ ] Filtre FIR testé (réponse en fréquence, délai de groupe)
- [ ] Export “bandpass.wav” généré

### 11.2. Détection/segmentation
- [ ] Courbes RMS/flux spectral tracées
- [ ] Seuil adaptatif validé (visuel + écoute)
- [ ] `segments.csv` rempli (>= 10 segments)

### 11.3. Features
- [ ] MFCC + Δ/ΔΔ extraits
- [ ] f0 et entropie calculés
- [ ] Fichier features sauvegardé

### 11.4. Évaluation
- [ ] Métriques calculées et commentées
- [ ] Figures (PR/F1/confusion) exportées
- [ ] Seed/config archivés

---

## 12. Annexe — Paramètres par défaut (point de départ)
- Fréquences de coupure : **400 Hz – 8 kHz** (ajuster selon spectrogramme)
- STFT : fenêtre **25 ms**, hop **10 ms**, FFT **1024**, fenêtre **Hann**
- Détection : médiane glissante 0,5–1 s, seuil **k = 3** (à balayer)
- MFCC : **20** coefficients, + Δ/ΔΔ
- f0 : portée **300–8 000 Hz**, méthode autocorrélation

---



