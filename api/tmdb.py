from typing import Optional

import httpx

from api.config import TMDB_READ_TOKEN

_TMDB_BASE = "https://api.themoviedb.org/3"
_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
_HEADERS = {"Authorization": f"Bearer {TMDB_READ_TOKEN}"}


async def fetch_movie_details(
    client: httpx.AsyncClient, tmdb_id: int
) -> dict[str, Optional[str]]:
    """Fetch poster URL and synopsis for a TMDB movie ID.

    Returns a dict with keys 'poster_url' and 'synopsis'.
    On any error (network, 404, …) returns both as None so the caller
    can still return a recommendation without TMDB data.
    """
    try:
        resp = await client.get(
            f"{_TMDB_BASE}/movie/{tmdb_id}",
            headers=_HEADERS,
            params={"language": "en-US"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        return {
            "poster_url": f"{_IMAGE_BASE}{poster_path}" if poster_path else None,
            "synopsis": data.get("overview") or None,
        }
    except Exception:
        return {"poster_url": None, "synopsis": None}
