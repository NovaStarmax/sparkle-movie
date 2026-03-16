# 🎬 Sparkle Movie — Système de recommandation de films avec Apache Spark

> Projet scolaire — RNCP Data / IA  
> Dataset : [MovieLens](https://grouplens.org/datasets/movielens/) (ml-latest-small & ml-32m)  
> Stack : PySpark 4.1.1 · Python 3.13 · Jupyter · Java 17

---

## Contexte

Une plateforme de streaming souhaite améliorer l'expérience utilisateur en proposant
des recommandations personnalisées. Ce projet implémente et compare trois approches
de recommandation sur le dataset MovieLens, en exploitant Apache Spark pour le
traitement distribué des données.

---

## Dataset

| Fichier       | Contenu                      | Taille (small) | Taille (large)    |
| ------------- | ---------------------------- | -------------- | ----------------- |
| `ratings.csv` | Notes utilisateurs (0.5–5.0) | 100 836 lignes | 32 000 204 lignes |
| `movies.csv`  | Métadonnées des films        | 9 742 films    | 87 585 films      |
| `tags.csv`    | Tags libres des utilisateurs | 3 683 lignes   | 2 000 072 lignes  |
| `links.csv`   | Identifiants IMDb / TMDb     | —              | —                 |

Les données couvrent la période janvier 1995 – octobre 2023, pour 200 948 utilisateurs
ayant chacun noté au minimum 20 films.

### Observations clés de l'exploration

- **Distribution des notes** : asymétrique vers les valeurs élevées, pic à 4.0.
  Biais de sélection attendu — les utilisateurs MovieLens sont des cinéphiles actifs.
- **Genres dominants** : Drama (34 175 films) et Comedy (23 123) représentent ~40%
  du catalogue. Film-Noir (353) et IMAX (195) sont très peu représentés.
- **Top films** : les documentaires TV (Planet Earth, Band of Brothers) dominent
  par note moyenne — MovieLens indexe films et séries sans distinction.
- **Qualité des données** : aucune valeur nulle, aucun doublon. 7 080 films sans
  genre (8%) exclus de l'approche content-based uniquement.

---

## Architecture du projet

```
sparkle-movie/
├── data/
│   ├── small/              # ml-latest-small (CSV sources)
│   ├── 32m/                # ml-32m (CSV sources)
│   └── processed/
│       ├── small/          # Parquet nettoyés
│       └── 32m/            # Parquet nettoyés
├── notebooks/
│   ├── 01_eda.ipynb        # Exploration, nettoyage, visualisations
│   └── 02_modelisation.ipynb  # ALS, Content-based, KNN, évaluation
├── outputs/                # Graphiques exportés
└── README.md
```

---

## Notebooks

### `01_eda.ipynb` — Exploration et nettoyage

1. Chargement des données avec schema explicite (Spark 4.x)
2. Audit qualité : nulls, doublons, plages de valeurs
3. Feature engineering : extraction de l'année, nettoyage des tags
4. Analyse exploratoire : top films, distribution des genres et des notes
5. Visualisations matplotlib
6. Persistance au format Parquet

### `02_modelisation.ipynb` — Modélisation et évaluation

1. **ALS** (Alternating Least Squares) — filtrage collaboratif matriciel
   - Grid search sur `rank` ∈ {5, 10, 15} × `regParam` ∈ {0.01, 0.1, 0.5}
   - Évaluation RMSE sur train/test split strict (80/20, seed=42)
2. **Content-based** — TF-IDF sur genres + tags, similarité cosinus
3. **KNN** — proximité utilisateurs par similarité cosinus
4. Évaluation comparative : Precision@10, Coverage
5. Recommandations pour 5 utilisateurs fictifs

---

## Algorithmes

### ALS — Alternating Least Squares

ALS factorise la matrice sparse users × films en deux matrices denses de dimension
`rank`. Chaque utilisateur et chaque film est représenté par un vecteur de facteurs
latents. La prédiction est le produit scalaire des deux vecteurs.

L'algorithme alterne : il fixe les vecteurs films et optimise les users, puis
l'inverse, jusqu'à convergence (`maxIter` itérations).

**Hyperparamètres optimaux** (grid search, 3-fold CV sur train) :

| Paramètre  | Valeur     |
| ---------- | ---------- |
| `rank`     | 5          |
| `regParam` | 0.1        |
| `maxIter`  | 10         |
| **RMSE**   | **0.8748** |

### Content-Based — TF-IDF + Similarité cosinus

Chaque film est décrit par ses genres et ses tags utilisateurs. On vectorise
ces descriptions via TF-IDF (HashingTF + IDF, `numFeatures=2048`), puis on
calcule la similarité cosinus entre films. Pour un utilisateur, on agrège
les similarités pondérées par ses notes sur les films aimés (≥ 4.0).

### KNN — K-Nearest Neighbors utilisateurs

Chaque utilisateur est représenté par un vecteur sparse de ses notes (dimensions
= films du catalogue). On calcule la similarité cosinus entre tous les pairs
d'utilisateurs, on garde les K=10 voisins les plus proches, et on recommande
les films bien notés par ces voisins que l'utilisateur cible n'a pas encore vus.

---

## Résultats

### Métriques comparatives (dataset small, 5 utilisateurs de test)

| Approche      | RMSE   | Precision@10 | Coverage (50 users) |
| ------------- | ------ | ------------ | ------------------- |
| ALS           | 0.8748 | 2.0%         | 1.3%                |
| KNN           | —      | **20.0%**    | 1.8%                |
| Content-based | —      | 4.0%         | **3.0%**            |

### Analyse

**KNN** obtient la meilleure précision (20%, soit ~100× le hasard) grâce à
la richesse des historiques utilisateurs sur le small dataset. Son principal
défaut est le biais de popularité et sa complexité quadratique en nombre
d'utilisateurs.

**Content-based** offre la meilleure couverture catalogue (3%) et résout
naturellement le cold start problem. Sa précision est limitée par la faible
densité des tags (65% des films sans tag).

**ALS** montre une précision faible sur le small dataset — ce qui est attendu.
Sa puissance s'exprime sur les grands volumes : sur ml-32m (32M ratings,
200k users), les facteurs latents capturent des patterns bien plus riches.
Son RMSE de 0.8748 reste néanmoins correct (écart moyen < 1 étoile).

### Recommandation finale

En production, un **système hybride** combinant les trois approches serait optimal :

```
score_final = α × score_ALS + β × score_KNN + γ × score_content_based
```

- ALS pour la personnalisation profonde (utilisateurs actifs)
- KNN pour les utilisateurs avec historique dense
- Content-based pour les nouveaux utilisateurs et films récents

---

## Installation et utilisation

### Prérequis

- Java JDK 17 (`java -version`)
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) comme gestionnaire de packages

### Installation

```bash
git clone https://github.com/NovaStarmax/sparkle-movie
cd sparkle-movie
uv sync
```

### Données

Télécharger les datasets depuis [GroupLens](https://grouplens.org/datasets/movielens/) :

- `ml-latest-small.zip` → extraire dans `data/small/`
- `ml-32m.zip` → extraire dans `data/32m/`

### Exécution

```bash
# Lancer Jupyter
uv run jupyter notebook

# Ouvrir dans l'ordre :
# 1. notebooks/01_eda.ipynb      (exécuter tout)
# 2. notebooks/02_modelisation.ipynb  (exécuter tout)
```

Pour basculer sur le large dataset, modifier dans chaque notebook :

```python
DATA = LARGE  # au lieu de DATA = SMALL
```

---

## Limites et perspectives

- **Évaluation sur 5 users** : statistiquement insuffisant. Une évaluation
  robuste nécessiterait 100+ utilisateurs avec leave-one-out cross-validation.
- **Cold start** : ALS et KNN ne peuvent pas recommander pour un nouvel
  utilisateur sans historique. Le content-based est la seule approche viable
  dans ce cas.
- **Scalabilité KNN** : O(n²) en nombre d'utilisateurs. Sur 200k users,
  une indexation par LSH (Locality-Sensitive Hashing) serait nécessaire.
- **Tags insuffisants** : 65% des films sans tag limitent la discrimination
  content-based. L'enrichissement via les synopsis (TMDb API) améliorerait
  significativement les résultats.

---

## Références

- Harper, F.M. & Konstan, J.A. (2015). _The MovieLens Datasets: History and Context_.
  ACM Transactions on Interactive Intelligent Systems, 5(4).
- [PySpark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [GroupLens Research](https://grouplens.org)
