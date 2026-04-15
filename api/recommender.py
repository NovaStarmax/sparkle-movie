import difflib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class Recommender:
    def __init__(self, models_path: Path) -> None:
        als_path = models_path / "als"
        shared_path = models_path / "shared"

        # Load ALS matrices — done once at startup
        self.user_ids: np.ndarray = np.load(als_path / "user_ids.npy")
        self.user_vectors: np.ndarray = np.load(als_path / "user_vectors.npy")
        self.item_ids: np.ndarray = np.load(als_path / "item_ids.npy")
        self.item_vectors: np.ndarray = np.load(als_path / "item_vectors.npy")

        with open(als_path / "metadata.json") as f:
            self.metadata: dict = json.load(f)

        # Load shared CSV data
        self.movies: pd.DataFrame = pd.read_csv(shared_path / "movies.csv")
        self.links: pd.DataFrame = pd.read_csv(shared_path / "links.csv")

        with open(shared_path / "movie_id_to_idx.json") as f:
            raw = json.load(f)
        self.movie_id_to_idx: dict[int, int] = {int(k): int(v) for k, v in raw.items()}

        # Precompute lowercased titles for fuzzy search
        self._titles_lower: list[str] = self.movies["title"].str.lower().tolist()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Autocomplete: return up to n movies whose title contains the query."""
        q = query.lower()
        mask = self.movies["title"].str.lower().str.contains(q, na=False, regex=False)
        return self.movies[mask].head(n)[["movieId", "title"]].to_dict("records")

    def find_movie(self, query: str) -> Optional[dict]:
        """Fuzzy-find a single movie by title (exact → contains → difflib)."""
        q = query.lower()

        # 1. Exact case-insensitive match
        exact = self.movies[self.movies["title"].str.lower() == q]
        if not exact.empty:
            return _row_to_dict(exact.iloc[0])

        # 2. Substring match — return the shortest title that contains the query
        contains = self.movies[
            self.movies["title"].str.lower().str.contains(q, na=False, regex=False)
        ]
        if not contains.empty:
            best = contains.iloc[contains["title"].str.len().argmin()]
            return _row_to_dict(best)

        # 3. Difflib fuzzy match
        close = difflib.get_close_matches(q, self._titles_lower, n=1, cutoff=0.6)
        if close:
            idx = self._titles_lower.index(close[0])
            return _row_to_dict(self.movies.iloc[idx])

        return None

    def recommend(self, movie_id: int, n: int = 10, top_users: int = 50) -> list[dict]:
        """Return top-n recommended movies for a given movieId using ALS vectors."""
        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return []

        item_vec = self.item_vectors[idx]  # (rank,)

        # Step 1 — find the top users who scored this item highest
        user_scores = self.user_vectors @ item_vec  # (n_users,)
        top_user_indices = np.argpartition(user_scores, -top_users)[-top_users:]

        # Step 2 — build mean user profile
        mean_profile = self.user_vectors[top_user_indices].mean(axis=0)  # (rank,)

        # Step 3 — score all items
        all_scores = self.item_vectors @ mean_profile  # (n_items,)

        # Exclude the input item
        all_scores[idx] = -np.inf

        # Step 4 — pick top n
        top_indices = np.argpartition(all_scores, -n)[-n:]
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]

        results: list[dict] = []
        for i in top_indices:
            item_movie_id = int(self.item_ids[i])
            movie_row = self.movies[self.movies["movieId"] == item_movie_id]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            genres_raw = row["genres"]
            results.append({
                "movieId": item_movie_id,
                "title": str(row["title"]),
                "genres": genres_raw.split("|") if pd.notna(genres_raw) else [],
                "predicted_score": float(all_scores[i]),
            })

        return results

    def get_tmdb_id(self, movie_id: int) -> Optional[int]:
        link = self.links[self.links["movieId"] == movie_id]
        if link.empty:
            return None
        val = link.iloc[0]["tmdbId"]
        return None if pd.isna(val) else int(val)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_dict(row: pd.Series) -> dict:
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row["genres"]) if pd.notna(row["genres"]) else "",
    }
