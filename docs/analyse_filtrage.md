##### Arthur DUFOUR, Ange SIMPALINGABO, Diego de RADIGUES

# Filtrage passe-bande


## Fréquence de coupure

La fréquence de coupure nous permet d'isoler le chant en supprimant les
bruits hauts et bas.

Les bruits indésirables bas sont généralement inférieurs à 400Hz.
Certains oiseaux, comme le canard ou l'oie, peuvent cependant chanter
autour de 200Hz.

Sachant que la plupart des oiseaux concentrent leurs cris autour de 1kHz
et 8kHz, la règle de bonne pratique serait d'établir la coupure basse à
400Hz et la coupure haute à 8kHz.

Si nous observons que le spectrogramme de l'oiseau semble plus aigu ou
plus grave, nous pourrons élargir le fenêtre de coupure vers le haut ou
vers le bas.

## Type de filtre numérique et ordre du filtre

Afin de conserver une phase linéaire et d'éviter la distorsion
temporelle très gênante dans l'analyse d'un chant d'oiseau, le filtre
numérique le plus approprié est le filtre à réponse impulsionnelle
finie.

La forme d'onde et les enveloppes du signal (pulsations, syllabes) sont
parfaitement conservées.

L'analyse se fera sur un fichier .wav donc pas en temps réel (cas qui
nécessite un filtrage à réponse impulsionnelle infinie).

------------------------------------------------------------------------

# Analyse en fréquence : paramètre ST(F)FT (Short Time (Fast) Fourier Transform)

Le signal sonore obtenu est d'abord continu dans le temps. Pour pouvoir
le traiter numériquement, il est nécessaire de l'échantillonner,
c'est-à-dire de le discrétiser en une suite de valeurs numériques.

On a besoin d'un signal numérique car les ordinateurs et
microcontrôleurs ne traitent que des nombres. Un signal continu
(analogique) ne peut pas être directement stocké, analysé ou filtré
numériquement. La discrétisation permet donc de le rendre traitable par
des algorithmes numériques.

## DFT 

Après avoir discrétisé le signal, on applique une DFT pour passer du
domaine temporel au domaine fréquentiel. Cela permet d'identifier les
fréquences présentes dans le signal et leur amplitude. On remarque que
la DFT donne les fréquences présentes et leurs amplitudes mais qu'elle
perd l'information temporelle.

<img width="1532" height="472" alt="DFT" src="https://github.com/user-attachments/assets/20435296-0bbf-4523-9a7c-81ba2705a184" />


## STFT

La STFT découpe le signal en petites fenêtres temporelles. On applique
une DFT sur chaque fenêtre ce qui permet de voir comment le contenu
fréquentiel évolue dans le temps. On choisit une fenêtre (ex. 20 ms de
signal) puis on la décale progressivement pour obtenir une carte
temps--fréquence.

La STFT repose sur le fenêtrage : on multiplie le signal par une fenêtre
(Hamming, Hann, etc.) pour isoler un petit segment, puis on applique la
DFT sur ce segment pour obtenir le spectre local.

<img width="904" height="832" alt="STFT" src="https://github.com/user-attachments/assets/c0b05712-804f-479d-b7d7-e14d466d7351" />


------------------------------------------------------------------------

## Paramètres essentiels du fenêtrage

-   **Taille de la fenêtre** : grande → bonne résolution en fréquence,
    faible résolution temporelle ; petite → bonne résolution temporelle,
    faible résolution fréquentielle.
-   **Taille de trame** : nombre d'échantillons dans chaque bloc
    analysé.
-   En pratique, fenêtre = trame est souvent suffisant.
-   On applique généralement un **recouvrement** pour éviter les
    discontinuités entre fenêtres.

    <img width="1180" height="734" alt="recouvrement" src="https://github.com/user-attachments/assets/56bbb07b-9639-4652-bcc0-4243752ed918" />


Formule :

    Recouvrement = 1 - (hop size / window size)

Le hop size fixe le décalage entre deux trames successives.

------------------------------------------------------------------------

Pour obtenir un spectrogramme de qualité :

-   Le choix de la taille de fenêtre détermine le compromis
    temps/fréquence.
-   Le hop size influence le recouvrement et la continuité du
    spectrogramme.
-   La fonction de fenêtrage réduit les discontinuités et améliore
    l'analyse.


