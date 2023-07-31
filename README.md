# Chess_AI_Pytorch
C'est un projet personnel qui vise à développer une Intelligence Artificielle (IA) pour le jeu d'échecs en utilisant PyTorch. L'objectif est de construire une IA qui, au lieu de se baser sur une fonction récursive min_max pour donner un score à une position, génère directement un coup à jouer. Pour la formation, nous utiliserons une base de données de tactiques de jeu lichess, qui contiens pour une position la suite de bons coups à jouer. Malgré une taille initiale relativement petite (500k), en 'explosant' le jeu de données (avec en moyenne 3.5 coups par tactique), nous atteignons un total ~3.5M de lignes.

## Modèle Inputs:
Ce modèle prend en entrée l'état complet de l'échiquier. Chaque pièce est représentée par sa position, et les couleurs sont encodées comme 1 et -1, basculant entre les pièces blanches et noires selon le trait de la position. Cela permet d'avoir seulement 378 entrées. Si nous avions créé des matrices séparées pour chaque couleur, nous aurions eu 786 entrées.

Quatre variantes de sortie de modèle seront testées dans ce projet :

## Modèle One Hot Encoder
L'entrée 'One Hot Encoder' vise à créer une représentation matricielle de l'échiquier. Cependant, au lieu de se baser sur une position complète, elle se concentre sur le mouvement qui doit être joué.

### Modèle One Hot Encoder 128x
Dans ce modèle, deux matrices de 64 sont combinées en une seule. La fonction d'activation est une fonction Sigmoid :

Une matrice de 64 pour déterminer quelle pièce doit être jouée,
Une autre matrice de 64 pour déterminer où cette pièce doit aller.

### Modèle One Hot Encoder 2 * 64x
Dans ce modèle, deux matrices de 64 sont séparées à la dernière couche cachée pour appliquer une fonction SoftMax sur chacune d'elles (car chaque matrice ne contient qu'un seul '1', comme dans une classification) :

- Une matrice de 64 pour déterminer quelle pièce doit être jouée,
- Une autre matrice de 64 pour déterminer où cette pièce doit aller.

## Modèle Auto Encoder
Dans ces modèles, la sortie est un espace latent d'un autoencodeur. M'objectif est de déterminer si un espace latent complexe est plus efficace qu'un one hot encoder qui ne contient que deux '1' et un reste de '0' (ou un seul '1' et un reste de '0' pour le '2*64x'). Comme il n'y a que 64 ou 4096 (64**2) entrées possibles, nous pouvons nous permettre d'avoir un surapprentissage puisque seul l'espace latent nous intéresse.

Remarque : La taille de l'espace latent sera choisie après avoir déterminé celle quelle est la plus optimisée.

### Modèle Auto Encoder 128x
Ce réseau de neurones prend en entrée et sorties les deux matrices One Hot Encoder et après l'entraînement, nous utilisons l'espace latent pour faire des prédictions. Quand notre modèle fera une prediction, nous traduirons l'espace latent (la sortie) en une matrice 128 one hot encoder le plus probable.

### Modèle Auto Encoder 2*64x
Ce réseau de neurones prend en entrée et sorties les deux matrices One Hot Encoder et après l'entraînement, nous utilisons l'espace latent pour faire des prédictions. Quand notre modèle fera une prediction, nous traduirons les 2 espaces latents (les sorties) en deux matrices 64 one hot encoder les plus probables.

## Améliorations :
Il faudra dans des versions futurs prendre en compte le rock, la prise en passant, qui sont pour l'instant jugée trop dur pour être appliqué à ce projet.

En fin de compte, 

Nous envisageons également d'étendre ce projet en développant un modèle capable de déterminer la meilleure position parmi deux positions données. Les processus de collecte et de vérification théoriques de leur qualité restent à préciser.
