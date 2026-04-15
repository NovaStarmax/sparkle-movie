# 🎬 Sparkle Movie

> Système de recommandation de films personnalisé, propulsé par Apache Spark et ALS.  
> Déployé sur [sparkle-movie.starspath-place.fr](https://sparkle-movie.starspath-place.fr)

---

## Contexte

Projet réalisé dans le cadre du MSc Big Data — EPITECH Marseille.

Une plateforme de streaming souhaite améliorer l'expérience utilisateur en proposant des recommandations personnalisées. Ce projet implémente et compare plusieurs approches de recommandation sur le dataset **MovieLens 32M** (32 millions de ratings, 200 948 utilisateurs, 87 585 films), en exploitant Apache Spark pour le traitement distribué des données.

---

## Démonstration live

🌐 **[sparkle-movie.starspath-place.fr](https://sparkle-movie.starspath-place.fr)**

Tapez un titre de film — l'algorithme ALS trouve les utilisateurs similaires et retourne 10 recommandations enrichies via l'API TMDB (affiches, synopsis).

**API Swagger** : [sparkle-movie.starspath-place.fr/docs](https://sparkle-movie.starspath-place.fr/docs)

---

## Dataset — MovieLens

Source : [GroupLens Research](https://grouplens.org/datasets/movielens/)

| Fichier | Contenu | Taille (small) | Taille (large) |
|---|---|---|---|
| `ratings.csv` | Notes utilisateurs (0.5–5.0) | 100 836 | 32 000 204 |
| `movies.csv` | Métadonnées des films | 9 742 films | 87 585 films |
| `tags.csv` | Tags libres utilisateurs | 3 683 | 2 000 072 |
| `links.csv` | Mapping MovieLens → IMDb / TMDb | — | — |

Le dossier `data/small/` est inclus dans le repo pour permettre de relancer les notebooks.  
Le dataset large doit être téléchargé depuis [grouplens.org](https://grouplens.org/datasets/movielens/).

---

## Architecture du projet

```
sparkle-movie/
├── api/                        # API FastAPI
│   ├── main.py                 # Routes + StaticFiles
│   ├── recommender.py          # Logique ALS (NumPy vectorisé)
│   ├── tmdb.py                 # Client TMDB async
│   ├── models.py               # Schémas Pydantic
│   └── config.py               # Variables d'environnement
├── frontend/
│   └── index.html              # SPA HTML/Tailwind/JS vanilla
├── models/
│   ├── als/                    # Matrices NumPy exportées depuis Spark
│   │   ├── user_vectors.npy    # (200 948, 5)
│   │   ├── item_vectors.npy    # (84 432, 5)
│   │   ├── user_ids.npy
│   │   ├── item_ids.npy
│   │   └── metadata.json
│   └── shared/                 # Métadonnées communes
│       ├── movies.csv
│       ├── links.csv
│       ├── movie_id_to_idx.json
│       └── ratings_count.csv
├── notebooks/
│   ├── 01_eda.ipynb            # Exploration et nettoyage
│   ├── 02_modelisation.ipynb   # ALS, Content-based, KNN (small)
│   ├── 03_modelisation_final.ipynb  # Modèles finaux (large)
│   ├── 04_neuMF.ipynb          # Neural Matrix Factorization
│   ├── 05_svdpp.ipynb          # SVD++ PyTorch
│   └── 06_export_matrices.ipynb     # Export NumPy pour l'API
├── tests/
│   ├── conftest.py             # Fixtures (mock si matrices absentes)
│   └── test_api.py             # 6 tests pytest
├── outputs/                    # Graphiques EDA
├── data/small/                 # Dataset small (tracké sur GitHub)
├── .github/workflows/ci.yml    # CI/CD GitHub Actions
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Installation

### Prérequis

- Python 3.13+
- Java JDK 17 (`java -version`)
- [uv](https://github.com/astral-sh/uv) — package manager

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Cloner et installer

```bash
git clone https://github.com/NovaStarmax/sparkle-movie
cd sparkle-movie
```

### Selon ce que vous voulez faire

```bash
# Lancer l'API uniquement
uv sync
uv run uvicorn api.main:app --reload

# Lancer les notebooks (EDA, modélisation, export)
uv sync --extra notebooks
uv run jupyter notebook

# Lancer les tests
uv sync --extra dev
uv run pytest tests/ -v

# Tout installer (développement complet)
uv sync --extra notebooks --extra dev
```

### Variables d'environnement

```bash
cp .env.example .env
# Remplir avec vos clés TMDB
```

```env
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_READ_TOKEN=your_tmdb_read_token_here
MODELS_PATH=models
```

Les clés TMDB sont disponibles gratuitement sur [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

---

## Algorithmes de recommandation

### ALS — Alternating Least Squares ⭐ Modèle en production

ALS factorise la matrice sparse users × films en deux matrices denses de dimension `rank` :

```
R ≈ U × Vᵀ    (200 948 users × 5 facteurs latents)
```

La prédiction est un produit scalaire : `r̂ᵤᵢ = uᵤ · vᵢ`

L'algorithme alterne deux étapes jusqu'à convergence :
1. Fixer les vecteurs films → optimiser les vecteurs users
2. Fixer les vecteurs users → optimiser les vecteurs films

**Hyperparamètres optimaux** (grid search, 3-fold CV) :

| Paramètre | Valeur |
|---|---|
| `rank` | 5 |
| `regParam` | 0.1 |
| `maxIter` | 10 |
| **RMSE (large)** | **0.8033** |

### Content-Based — TF-IDF + Similarité cosinus

Chaque film est décrit par ses genres et tags utilisateurs, vectorisé via TF-IDF (`numFeatures=2048`). La similarité cosinus identifie les films au contenu proche.

Couverture catalogue : **3.0%** sur 50 users — meilleure couverture, ne nécessite pas d'historique de ratings.

### LSH-KNN — K-Nearest Neighbors avec Locality-Sensitive Hashing

Chaque utilisateur est représenté par un vecteur sparse de ses ratings. `BucketedRandomProjectionLSH` de Spark MLlib approxime les K voisins les plus proches en O(n log n) au lieu de O(n²).

**Precision@10 : 20%** sur le small dataset (~100× mieux que le hasard).

### NeuMF — Neural Matrix Factorization (expérimental)

Implémentation PyTorch de He et al. (2017). Combine GMF (produit de Hadamard) et MLP pour capturer des interactions non-linéaires.

**Résultat : RMSE 0.8956** — inférieur à ALS sur ce dataset. Les réseaux de neurones montrent leur supériorité sur les feedbacks implicites (clics, vues), pas sur les notes explicites.

### SVD++ (expérimental)

Extension d'ALS avec biais utilisateur/film et signal implicite (Koren, 2009 — Netflix Prize).
Implémenté from scratch en PyTorch suite à l'incompatibilité de `scikit-surprise` avec Python 3.13.

**Résultat : RMSE 0.8658** — meilleur que NeuMF, pas encore meilleur qu'ALS faute de tuning complet.

---

## Résultats comparatifs

| Modèle | RMSE | Precision@10 | Coverage | Dataset |
|---|---|---|---|---|
| **ALS** | **0.8033** | — | 0.1% | 32M ratings |
| LSH-KNN | — | **20.0%** | 0.3% | 100k ratings |
| Content-based | — | 4.0% | **3.0%** | 100k ratings |
| NeuMF | 0.8956 | — | — | 10M ratings |
| SVD++ | 0.8658 | — | — | 100k ratings |

**Conclusion** : ALS reste la référence sur les ratings explicites MovieLens. Sa puissance s'exprime pleinement sur le volume complet (32M ratings). En production, un système hybride `α×ALS + β×KNN + γ×content` serait optimal.

---

## API REST

Base URL : `https://sparkle-movie.starspath-place.fr`

### `GET /recommend/movie/{movie_title}`

Retourne 10 films recommandés pour un film donné, enrichis via TMDB.

```bash
curl https://sparkle-movie.starspath-place.fr/recommend/movie/Toy%20Story
```

```json
{
  "query": "Toy Story",
  "matched_movie": {
    "movieId": 1,
    "title": "Toy Story (1995)",
    "tmdb_id": 862,
    "poster_url": "https://image.tmdb.org/t/p/w500/...",
    "synopsis": "..."
  },
  "recommendations": [
    {
      "movieId": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": ["Crime", "Drama"],
      "predicted_score": 5.83,
      "tmdb_id": 278,
      "poster_url": "https://image.tmdb.org/t/p/w500/...",
      "synopsis": "...",
      "release_year": 1994
    }
  ]
}
```

### `GET /search?q={query}`

Autocomplétion — retourne les 5 films correspondant à la recherche.

```bash
curl "https://sparkle-movie.starspath-place.fr/search?q=toy"
```

### `GET /health`

Health check — statut de l'API et informations sur le modèle chargé.

```bash
curl https://sparkle-movie.starspath-place.fr/health
```

```json
{
  "status": "ok",
  "model": "ALS",
  "n_users": 200948,
  "n_items": 84432,
  "rmse": 0.8033
}
```

### `GET /docs`

Swagger UI — documentation interactive complète de l'API.

---

## APIs tierces utilisées

### TMDB — The Movie Database

Enrichissement des recommandations avec affiches et synopsis.

- **Endpoint** : `https://api.themoviedb.org/3/movie/{tmdb_id}`
- **Auth** : Bearer token (`TMDB_READ_TOKEN`)
- **Graceful degradation** : si TMDB est indisponible, l'API retourne les recommandations sans poster ni synopsis
- **Mapping** : `links.csv` fait le pont entre `movieId` (MovieLens) et `tmdb_id` (TMDB)

Clés gratuites disponibles sur [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

---

## Déploiement

### Docker

```bash
docker build -t sparkle-movie .
docker-compose up
```

L'image multi-stage fait ~225 MB. Les dépendances notebooks (PySpark, PyTorch) sont exclues de l'image de production.

### CI/CD — GitHub Actions + Coolify

```
Push sur master
    → GitHub Actions : uv sync + pytest (6 tests)
    → Si vert → webhook Coolify
    → Coolify : docker build + rolling update
    → App live sur sparkle-movie.starspath-place.fr
```

Les matrices NumPy sont montées en volume read-only depuis `/data/sparkle-movie/models` sur le VPS — elles ne passent pas par GitHub (trop lourdes).

---

## Tests

```bash
uv run pytest tests/ -v
```

| Test | Description |
|---|---|
| `test_health` | `/health` retourne status ok |
| `test_search` | `/search?q=toy` retourne des résultats |
| `test_recommend_valid` | `/recommend/movie/Toy Story` retourne 10 films |
| `test_recommend_invalid` | Film inconnu → 404 |
| `test_recommend_performance` | Réponse < 2 secondes (hors TMDB) |
| `test_recommend_quality` | Tous les films recommandés ont ≥ 50 ratings |

Les tests utilisent des matrices mockées si les vraies matrices sont absentes (CI sans accès aux fichiers lourds).

---

## État de l'art — Justification des choix techniques

### Pourquoi Apache Spark ?

MovieLens 32M contient 32 millions de ratings. Pandas chargerait ce fichier en ~2 Go de RAM avec des opérations mono-thread. Spark distribue les calculs sur tous les cœurs disponibles et traite les données de façon lazy — seuls les calculs nécessaires sont exécutés.

Sur ce projet (machine locale M4 Pro, 24 Go RAM) : Spark permet d'entraîner ALS sur 32M ratings en ~45 minutes là où une implémentation NumPy naïve serait prohibitive.

### Pourquoi ALS plutôt que Deep Learning ?

ALS est précisément conçu pour la factorisation de matrices de ratings explicites. Les réseaux de neurones (NeuMF, SVD++) montrent leur supériorité sur les feedbacks implicites (clics, vues) et nécessitent des volumes de données bien plus importants pour converger.

Sur MovieLens avec notes explicites : **ALS (0.8033) > SVD++ (0.8658) > NeuMF (0.8956)**.

### Pourquoi NumPy pour le serving ?

Démarrer une SparkSession prend 30 secondes — incompatible avec une API temps réel. Les matrices ALS (< 10 Mo) sont chargées une fois en mémoire au démarrage de l'API. La prédiction est un produit matriciel NumPy vectorisé : `scores = user_vector @ item_vectors.T` — microseconde par requête.

### Conformité RGPD

MovieLens est un dataset de recherche anonymisé — aucune donnée personnelle identifiable. En production, un système de recommandation personnalisé nécessiterait :
- Consentement explicite de l'utilisateur (opt-in)
- Droit à l'oubli — suppression des vecteurs latents associés à un userId
- Minimisation des données — ne collecter que les ratings nécessaires
- Base légale : intérêt légitime ou consentement selon le contexte RGPD

---

## Références

- Harper, F.M. & Konstan, J.A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS 5(4).
- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW 2017.
- Koren, Y. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- Zhou, Y. et al. (2008). *Large-Scale Parallel Collaborative Filtering for the Netflix Prize*. AAIM 2008.

---

## Auteur

**Antoine Gobbe** — MSc Big Data, EPITECH Marseille  
Alternance : Chargé de mission Data & IA — Office de Tourisme et des Congrès de Marseille