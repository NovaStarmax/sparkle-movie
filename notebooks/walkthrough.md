# Sparkle Movie — Démarche technique détaillée

Ce document explique pas à pas la logique de chaque décision technique prise
dans les notebooks `01_eda.ipynb` et `03_final_models.ipynb`.

---

## 1. Environnement et session Spark

### Pourquoi Java ?

PySpark est un wrapper Python autour d'Apache Spark, qui est écrit en Java/Scala.
Python ne peut pas exécuter Spark directement — il communique avec la JVM via
une couche appelée **Py4J**. Java doit donc être installé et accessible avant
tout lancement de PySpark. La version 17 (LTS) est utilisée ici car elle est
compatible avec Spark 4.x.

### Configuration de la SparkSession

```python
spark = SparkSession.builder \
    .appName("SparkleMovie-Final") \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.python.worker.reuse", "true") \
    .getOrCreate()
```

Chaque paramètre a une justification précise :

- **`local[*]`** : Spark tourne en mode local (pas de cluster), en utilisant
  tous les cœurs CPU disponibles. Le `*` est dynamique — sur un M4 Pro à 12 cœurs,
  Spark lance 12 threads de calcul parallèles.
- **`spark.driver.memory = 16g`** : Le driver est le processus Python principal
  qui orchestre les calculs. Sur le large dataset (32M ratings), les opérations
  de collect() peuvent rapatrier plusieurs Go de données vers le driver.
- **`spark.driver.maxResultSize = 4g`** : Limite la taille d'un seul résultat
  collecté. Évite les OOM (Out Of Memory) silencieux.
- **`spark.sql.shuffle.partitions = 200`** : Par défaut Spark crée 200 partitions
  lors des shuffles (groupBy, join). Sur une machine locale à 12 cœurs, 200 est
  un bon compromis entre parallélisme et overhead.
- **`spark.python.worker.reuse = true`** : Réutilise les processus Python workers
  au lieu d'en créer un nouveau par tâche. Réduit la consommation de sockets
  réseau — crucial pour éviter le `No buffer space available` sur macOS.

### Pourquoi `getOrCreate()` et pas `build()` ?

`getOrCreate()` retourne la session existante si elle est déjà active, ou en
crée une nouvelle. Dans un notebook, on peut relancer une cellule sans crasher
("une session est déjà active"). `build()` n'existe pas — c'est le pattern
Builder qui accumule les configs, et `getOrCreate()` en est le terminal.

---

## 2. Chargement des données avec schema explicite

### Pourquoi définir le schema manuellement ?

```python
ratings_schema = StructType([
    StructField("userId",    IntegerType(), nullable=False),
    StructField("movieId",   IntegerType(), nullable=False),
    StructField("rating",    FloatType(),   nullable=False),
    StructField("timestamp", LongType(),    nullable=False),
])
```

Avec `inferSchema=True`, Spark scanne tout le fichier CSV pour deviner les types.
Sur 32M lignes, ce scan prend plusieurs minutes et consomme des ressources inutilement.
En définissant le schema manuellement, Spark lit directement avec les bons types.

Choix des types :
- `IntegerType` pour userId/movieId : jamais de virgule, jamais de texte.
- `FloatType` pour rating : les notes ont des demi-étoiles (0.5, 3.5...).
- `LongType` pour timestamp : les secondes depuis 1970 dépassent la limite
  d'un Int32 (2 147 483 647 = année 2038). Un Long64 est nécessaire.

### Comportement de nullable en Spark 4.x

Même avec `nullable=False`, Spark affiche `nullable = true` dans le schema
sur les sources CSV. C'est un comportement connu : Spark ne peut pas garantir
l'absence de valeurs vides dans un CSV à la lecture. Le `nullable=False` est
une intention déclarative, pas une contrainte appliquée. C'est pourquoi on
ajoute une étape de nettoyage explicite avec `dropna()`.

---

## 3. Nettoyage des données

### Audit qualité

```python
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])
```

Cette expression génère dynamiquement une colonne de comptage de nulls pour
chaque colonne du DataFrame. C'est plus concis qu'un `filter().count()` par colonne
et s'exécute en un seul passage sur les données (un seul job Spark).

### Décision sur les films sans genre

7 080 films (8% du large dataset) ont `genres = "(no genres listed)"`. Décision :
**ne pas les supprimer du dataset principal**. ALS et KNN se basent uniquement
sur les ratings — le genre n'intervient pas dans leur calcul. Supprimer ces films
appauvrirait inutilement les recommandations collaboratives.

On crée un sous-ensemble `movies_with_genres` réservé au content-based uniquement.
C'est la décision la plus conservative : on ne perd aucune information.

### Extraction de l'année

```python
movies_clean = movies_clean.withColumn(
    "year",
    when(
        regexp_extract(col("title"), r"\((\d{4})\)$", 1) != "",
        regexp_extract(col("title"), r"\((\d{4})\)$", 1).cast("integer")
    ).otherwise(None)
)
```

Le pattern `\((\d{4})\)$` capture 4 chiffres entre parenthèses en fin de titre.
En Spark 4.x, `regexp_extract` retourne une chaîne vide `""` (et non `null`)
quand il n'y a pas de match. Un `.cast("integer")` direct sur `""` lève une
`NumberFormatException`. On utilise donc `when(...!= "")` pour intercepter
ce cas avant le cast. C'est une incompatibilité introduite dans Spark 4.x
par rapport aux versions antérieures.

### Sauvegarde en Parquet

```python
ratings_clean.write.parquet(f"{PATH}ratings_clean.parquet", mode="overwrite")
```

Le format Parquet présente trois avantages par rapport au CSV :
1. **Schema préservé** : les types (IntegerType, FloatType...) sont encodés
   dans le fichier. Plus besoin de les redéfinir à chaque lecture.
2. **Compression columaire** : Parquet stocke les données par colonne, pas par
   ligne. Sur `ratings.csv`, la colonne `rating` (valeurs répétitives comme
   3.0, 4.0, 4.5) se compresse très efficacement.
3. **Lecture partielle** : on peut lire uniquement les colonnes nécessaires
   sans scanner tout le fichier.

Sur le large dataset, le fichier `ratings_clean.parquet` est ~3x plus petit
que le CSV source et se relit ~5x plus vite.

---

## 4. ALS — Alternating Least Squares

### Principe mathématique

ALS factorise la matrice sparse users × films **R** (610 users × 9 742 films
sur le small, 200k × 87k sur le large) en deux matrices denses :

```
R ≈ U × Vᵀ
```

Où :
- **U** est la matrice users (n_users × rank)
- **V** est la matrice films (n_films × rank)
- **rank** est la dimension des facteurs latents (5 dans notre cas optimal)

Chaque ligne de U est le vecteur d'un utilisateur, chaque ligne de V est le
vecteur d'un film. La prédiction de note pour l'user i sur le film j est :

```
r̂ᵢⱼ = uᵢ · vⱼ  (produit scalaire)
```

L'algorithme "Alternating" alterne deux étapes jusqu'à convergence :
1. Fixer V, optimiser U (régression linéaire par utilisateur)
2. Fixer U, optimiser V (régression linéaire par film)

Chaque étape est une régression linéaire avec solution analytique exacte
(inverse de matrice). C'est pourquoi ALS est parallélisable : chaque
utilisateur/film est indépendant des autres à chaque étape.

### Hyperparamètres et leur impact

**`rank=5`** : nombre de facteurs latents. Sur le small dataset (610 users),
un rank élevé surfit : il y a plus de dimensions que de patterns réels dans
les données. Le grid search a confirmé que rank=5 généralise mieux que rank=10
ou rank=15. Sur le large (200k users), un rank plus élevé serait justifié.

**`regParam=0.1`** : coefficient de régularisation L2. Sans régularisation,
ALS tend à surfit sur les utilisateurs avec peu de ratings (un user avec 3 notes
peut avoir des facteurs latents très extrêmes qui s'effondrent sur le test set).
La régularisation pénalise les vecteurs de grande norme.

**`coldStartStrategy="drop"`** : si un userId ou movieId du test set n'apparaît
pas dans le train (parce que le split aléatoire l'a entièrement mis en test),
ALS ne peut pas calculer de prédiction. `"drop"` supprime ces lignes plutôt que
de retourner `NaN` qui fausserait le calcul du RMSE.

### Clipping des prédictions

```python
.withColumn("prediction", greatest(lit(0.5), least(lit(5.0), col("prediction"))))
```

ALS est un modèle de factorisation matricielle sans contrainte de bornes.
Le produit scalaire de deux vecteurs peut théoriquement être n'importe quel
réel — on a observé des prédictions jusqu'à 6.03 sur certains films.
Le clipping force les prédictions dans la plage valide [0.5, 5.0].

### Grid search sans data leakage

```python
cv_model = cv.fit(train)  # train uniquement, pas ratings_clean
```

Une erreur classique consiste à passer `ratings_clean` au CrossValidator.
Spark MLlib entraîne alors sur le dataset complet avec une validation croisée
interne — le test set devient visible pendant le tuning. En passant `train`
uniquement, le test set reste invisible pendant toute la phase de tuning.
Cette correction a réduit le RMSE apparent de 0.66 (optimiste) à 0.87 (réel).

---

## 5. Content-Based — TF-IDF et similarité cosinus

### Construction du champ content

```python
movies_with_content = movies_with_content.withColumn(
    "content",
    concat_ws(" ",
        regexp_replace(col("genres"), "\\|", " "),
        col("tags_text")
    )
)
```

Le champ `genres` contient des valeurs pipe-séparées (`"Action|Comedy|Drama"`).
On remplace `|` par des espaces pour obtenir une chaîne tokenizable
(`"Action Comedy Drama"`). On concatène ensuite les tags utilisateurs agrégés
par film. Le résultat final est un "document textuel" par film :
`"Adventure Animation Children Comedy Fantasy pixar fun animated"`.

### Pipeline TF-IDF

```python
tokenizer = Tokenizer(inputCol="content", outputCol="words")
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2048)
idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2)
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
```

**Étape 1 — Tokenization** : découpe la chaîne en liste de mots.
`"Action Comedy Drama"` → `["action", "comedy", "drama"]`

**Étape 2 — HashingTF (Term Frequency)** : transforme la liste de mots en
vecteur numérique. Chaque mot est hashé vers un index dans un espace de
dimension `numFeatures=2048`. La valeur à cet index est la fréquence du mot
dans le document. `numFeatures=2048` est un compromis : trop petit = collisions
de hash (deux mots différents mappés au même index), trop grand = vecteurs
très sparse consommant beaucoup de mémoire.

**Étape 3 — IDF (Inverse Document Frequency)** : pondère les fréquences par
la rareté du terme. Un terme présent dans tous les films (comme "Drama") a
un IDF faible — il est peu discriminant. Un terme rare (comme "existentialism")
a un IDF élevé.

`minDocFreq=2` : ignore les termes qui n'apparaissent que dans un seul film.
Ces termes ultra-rares (souvent des erreurs de frappe dans les tags) ont un
IDF artificiellement élevé qui perturberait les similarités.

**Formule TF-IDF** : `tfidf(t, d) = tf(t, d) × log(N / df(t))`
où N = nombre total de documents, df(t) = nombre de documents contenant t.

### Similarité cosinus

```python
def cosine_similarity(v1, v2):
    v1_dense = np.array(v1.toArray())
    v2_dense = np.array(v2.toArray())
    norm1 = np.linalg.norm(v1_dense)
    norm2 = np.linalg.norm(v2_dense)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1_dense, v2_dense) / (norm1 * norm2))
```

La similarité cosinus mesure l'angle entre deux vecteurs, indépendamment
de leur magnitude. Un film avec 3 genres et un film avec 8 genres peuvent
avoir une similarité élevée si leurs directions sont proches.

```
cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
```

La valeur est comprise entre -1 et 1. En pratique sur des vecteurs TF-IDF
(toutes valeurs positives), elle est entre 0 et 1.

**Pourquoi `.toArray()` ?** Les vecteurs Spark sont des `SparseVector` — ils
ne stockent que les valeurs non-nulles pour économiser de la mémoire. NumPy
ne peut pas calculer directement sur des SparseVector. `.toArray()` convertit
en array dense de dimension `numFeatures=2048`.

### Agrégation des scores par utilisateur

```python
scores[mid]["score"] += sim * row.rating
```

Pour un utilisateur donné, on ne prend pas juste le film le plus similaire à
un seul film aimé. On agrège les similarités de tous les films aimés, pondérées
par la note donnée. Un film très similaire à plusieurs films bien notés par
l'utilisateur obtient un score composite élevé. C'est une forme de
**profil utilisateur implicite** : la somme de ses préférences passées.

---

## 6. LSH-KNN — K-Nearest Neighbors avec Locality-Sensitive Hashing

### Pourquoi pas le KNN naïf ?

Le KNN naïf calcule la similarité entre chaque paire d'utilisateurs.
Sur N utilisateurs, cela représente N×(N-1)/2 comparaisons :

- Small (610 users) : ~186 000 comparaisons → 0.3 secondes ✓
- Large (200 948 users) : ~20 milliards de comparaisons → plusieurs jours ✗

La complexité O(n²) est fondamentalement non-scalable. C'est une limite
algorithmique, pas une limite matérielle.

### Principe de LSH

`BucketedRandomProjectionLSH` projette les vecteurs utilisateurs dans un espace
de dimension réduite via des projections aléatoires. Les utilisateurs similaires
(vecteurs proches) tombent dans le même "bucket" avec haute probabilité.
On ne compare que les utilisateurs du même bucket — complexité O(n log n).

**Paramètres :**
- `bucketLength=2.0` : largeur d'un bucket. Plus petit = buckets plus fins
  = moins de faux positifs mais plus de faux négatifs.
- `numHashTables=3` : nombre de tables de hachage indépendantes. Plus de tables
  = meilleur recall (moins de vrais voisins manqués) mais plus de mémoire.

**Compromis scalabilité/précision** : `approxNearestNeighbors` retourne des
voisins *approximatifs* — pas nécessairement les K plus proches exacts.
C'est le prix de la scalabilité. En pratique sur des datasets réels,
la dégradation de précision est faible (< 5%) pour `numHashTables >= 3`.

### Réduction dimensionnelle préalable

Les vecteurs utilisateurs ont initialement 87 585 dimensions (un axe par film).
Même sparse, 10 000 vecteurs de 87k dimensions dépassent les capacités mémoire
d'une machine locale pour le LSH fit.

Solution : ne garder que les films ayant ≥ 50 ratings. Ces ~5 000 films
"populaires" couvrent l'écrasante majorité des ratings (loi de Pareto :
5% des films concentrent 80% des ratings). Les vecteurs passent de
87k → ~5k dimensions — réduction de 17x sans perte significative d'information.

### Conversion distance → similarité

```python
similarity = 1 / (1 + distance)
```

`approxNearestNeighbors` retourne une **distance euclidienne** (pas une
similarité). On la convertit en similarité via `1 / (1 + d)` :
- Distance = 0 → similarité = 1.0 (utilisateurs identiques)
- Distance = 1 → similarité = 0.5
- Distance → ∞ → similarité → 0

Cette transformation monotone décroissante préserve l'ordre de ranking
tout en normalisant dans [0, 1].

---

## 7. Évaluation

### RMSE (Root Mean Square Error)

```
RMSE = sqrt(1/n × Σ(r̂ᵢ - rᵢ)²)
```

Applicable uniquement à ALS car c'est le seul algorithme qui prédit des
valeurs numériques de notes. RMSE de 0.8033 sur le large : en moyenne,
la prédiction s'écarte de 0.80 étoile de la note réelle.

**Interprétation contextualisée** : l'écart-type des notes MovieLens est ~1.1.
Un RMSE de 0.80 est inférieur à cet écart-type — le modèle prédit mieux
que la simple moyenne globale (baseline naïve).

### Precision@K

```
Precision@K = |films recommandés ∩ films aimés dans le test| / K
```

Mesure la proportion de "bonnes" recommandations parmi les K proposées.
Un film est "aimé" si sa note dans le test set est ≥ 4.0.

**Pourquoi Precision@K = 0 pour ALS sur le large ?**
Le test set d'un utilisateur contient ~20% de ses ratings — souvent 50-200
films sur 87 585 disponibles. ALS recommande 10 films très spécifiques et
souvent peu populaires. La probabilité de tomber exactement dans les 50-200
films du test set parmi 87k est quasi nulle. Ce n'est pas un échec d'ALS —
c'est un désalignement entre la métrique Precision@K et le comportement
d'un algorithme qui recommande des contenus rares.

### Coverage

```
Coverage = |films recommandés distincts| / |catalogue total|
```

Mesure la diversité des recommandations à travers les utilisateurs.
Les faibles valeurs (0.1% à 0.3%) s'expliquent par le calcul sur seulement
50 users × 10 films = 500 recommandations sur 87k films. Ce chiffre serait
significativement plus élevé sur l'ensemble des 200k utilisateurs.

---

## 8. Décisions architecturales

### Deux notebooks, pas un

Le sujet imposait "au maximum 2 notebooks". L'architecture retenue :
- `01_eda.ipynb` : exploration, nettoyage, visualisations, sauvegarde Parquet
- `03_final_models.ipynb` : modélisation sur le large dataset

Cette séparation a un avantage pratique majeur : si la modélisation crash
(ce qui arrive fréquemment sur 32M ratings), on n'a pas besoin de relancer
l'exploration depuis le début. Les Parquet sauvegardés à l'étape 1 sont
rechargés directement.

### Variables DATA / SMALL / LARGE

```python
SMALL = "../data/processed/small/"
LARGE = "../data/processed/32m/"
DATA  = LARGE
```

Ce pattern permet de basculer entre datasets en changeant une seule variable.
Tout le reste du code est agnostique à la taille du dataset. Cette architecture
est une bonne pratique de data engineering : le code ne doit pas contenir de
chemins hardcodés.

### Parquet plutôt que CSV

La pipeline complète lit les CSV sources une seule fois (dans `01_eda.ipynb`),
nettoie les données, et sauvegarde en Parquet. Toutes les étapes suivantes
lisent les Parquet. Sur le large dataset, cette décision économise environ
5 minutes de lecture à chaque relancement du notebook de modélisation.

---

## 9. Limites et perspectives

**Évaluation** : la Precision@K sur 5 utilisateurs n'est pas représentative.
Une évaluation rigoureuse utiliserait leave-one-out cross-validation sur
100+ utilisateurs.

**LSH sur 10k users** : l'échantillonnage à 5% introduit un biais. Un cluster
Spark (même petit — 3 nœuds) permettrait de traiter les 200k users complets.

**Enrichissement content-based** : les synopsis TMDb (disponibles via API)
amélioreraient significativement la qualité des vecteurs TF-IDF, notamment
pour les films récents peu tagués par les utilisateurs.

**Système hybride** : aucune approche seule n'est optimale. La combinaison
`α×ALS + β×LSH-KNN + γ×content-based` avec des coefficients calibrés par
profil utilisateur est l'architecture standard des plateformes de streaming
en production.