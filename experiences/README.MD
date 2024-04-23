# Possibilités des expériences
## Entrées du modèle prédictif
* 768 entrées (64 cases pour 6 pièces, pour les 2 couleurs, vide = 0, existante = 1)
* 384 sorties (64 cases pour 12 pièces, vide = 0, blanc = 1, noir = -1)

## Sorties du modèle prédictif
### Encodage à chaud
* 128 sorties (64 cases pour la pièce qui va bouger, 64 cases pour la case d'arrivée) **Sigmoid**
* 2x 64 sorties (64 cases pour la pièce qui va bouger, 64 cases pour la case d'arrivée) **SoftMax**

### Espace lattent (Auto Encodeur)
#### 1x Espace lattent
On entraine l'auto encodeur pour transformer une matrice de 128 représentants un coup entier (pièce à déplacer et son futur emplacement)
* Petite taille (8 entrées)
* Moyenne taille (16 entrées)
* Grande taille (32 entrées)

#### 2x Espace lattent
On entraine l'auto encodeur pour transformer une matrice de 64 représentants soit une pièce par son emplacement, soit son futur emplacement, et le modèle prédira 2 espaces latent
*  Petite taille (4 entrées)
*  Moyenne taille (8 entrées)
*  Grande taille (16 entrées)

## Affinage du modèle
* Le modèle n'est pas pré-entrainé et apprendre directement à jouer les meilleurs coups
* Le modèle est pré-entrainé à jouer des coups possibles sur des positions aléatoires mais réalistes

-----------------

# Liste des expériences
## Avec 768 entrées
### Expérience-1
768 entrées, modèle non pré-entraîné, encodage à chaud, 128 sorties avec 2x SoftMax.
### Expérience-2
768 entrées, modèle non pré-entraîné, encodage à chaud, 2x 64 sorties avec SoftMax.
### Expérience-3
768 entrées, modèle non pré-entraîné, 1x espace lattent de petite taille (8 entrées).
### Expérience-4
768 entrées, modèle non pré-entraîné, 1x espace lattent de moyenne taille (16 entrées).
### Expérience-5
768 entrées, modèle non pré-entraîné, 1x espace lattent de grande taille (32 entrées).
### Expérience-6
768 entrées, modèle non pré-entraîné, 2x espace lattent de petite taille (4 entrées).
### Expérience-7
768 entrées, modèle non pré-entraîné, 2x espace lattent de moyenne taille (8 entrées).
### Expérience-8
768 entrées, modèle non pré-entraîné, 2x espace lattent de grande taille (16 entrées).
### Expérience-9
768 entrées, modèle pré-entraîné, encodage à chaud, 128 sorties avec Sigmoid.
### Expérience-10
768 entrées, modèle pré-entraîné, encodage à chaud, 2x 64 sorties avec SoftMax.
### Expérience-11
768 entrées, modèle pré-entraîné, 1x espace lattent de petite taille (8 entrées).
### Expérience-12
768 entrées, modèle pré-entraîné, 1x espace lattent de moyenne taille (16 entrées).
### Expérience-13
768 entrées, modèle pré-entraîné, 1x espace lattent de grande taille (32 entrées).
### Expérience-14
768 entrées, modèle pré-entraîné, 2x espace lattent de petite taille (4 entrées).
### Expérience-15
768 entrées, modèle pré-entraîné, 2x espace lattent de moyenne taille (8 entrées).
### Expérience-16
768 entrées, modèle pré-entraîné, 2x espace lattent de grande taille (16 entrées).

## Avec 384 entrées
### Expérience-17
384 entrées, modèle non pré-entraîné, encodage à chaud, 128 sorties avec Sigmoid.
### Expérience-18
384 entrées, modèle non pré-entraîné, encodage à chaud, 2x 64 sorties avec SoftMax.
### Expérience-19
384 entrées, modèle non pré-entraîné, 1x espace lattent de petite taille (8 entrées).
### Expérience-20
384 entrées, modèle non pré-entraîné, 1x espace lattent de moyenne taille (16 entrées).
### Expérience-21
384 entrées, modèle non pré-entraîné, 1x espace lattent de grande taille (32 entrées).
### Expérience-22
384 entrées, modèle non pré-entraîné, 2x espace lattent de petite taille (4 entrées).
### Expérience-23
384 entrées, modèle non pré-entraîné, 2x espace lattent de moyenne taille (8 entrées).
### Expérience-24
384 entrées, modèle non pré-entraîné, 2x espace lattent de grande taille (16 entrées).
### Expérience-25
384 entrées, modèle pré-entraîné, encodage à chaud, 128 sorties avec Sigmoid.
### Expérience-26
384 entrées, modèle pré-entraîné, encodage à chaud, 2x 64 sorties avec SoftMax.
### Expérience-27
384 entrées, modèle pré-entraîné, 1x espace lattent de petite taille (8 entrées).
### Expérience-28
384 entrées, modèle pré-entraîné, 1x espace lattent de moyenne taille (16 entrées).
### Expérience-29
384 entrées, modèle pré-entraîné, 1x espace lattent de grande taille (32 entrées).
### Expérience-30
384 entrées, modèle pré-entraîné, 2x espace lattent de petite taille (4 entrées).
### Expérience-31
384 entrées, modèle pré-entraîné, 2x espace lattent de moyenne taille (8 entrées).
### Expérience-32
384 entrées, modèle pré-entraîné, 2x espace lattent de grande taille (16 entrées).

Cela donne un total de 32 expériences possibles, en tenant compte des différentes options d'entrée, du type d'entraînement, du type de sortie et de la taille de l'espace lattent.