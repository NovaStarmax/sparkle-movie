import csv
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

_RATINGS_COUNT_PATH = Path(__file__).parent.parent / "models" / "shared" / "ratings_count.csv"


@pytest.fixture(scope="module")
async def client(setup_recommender):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "ALS"
    assert data["n_users"] > 0
    assert data["n_items"] > 0
    assert isinstance(data["rmse"], float)


async def test_search(client):
    resp = await client.get("/search?q=toy")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "movieId" in data[0]
    assert "title" in data[0]


async def test_recommend_valid(client):
    resp = await client.get("/recommend/movie/Toy%20Story")
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "matched_movie" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 10
    rec = data["recommendations"][0]
    for field in ("movieId", "title", "genres", "predicted_score"):
        assert field in rec, f"Missing field: {field}"
    assert isinstance(rec["genres"], list)


async def test_recommend_invalid(client):
    resp = await client.get("/recommend/movie/xyzabc")
    assert resp.status_code == 404


async def test_recommend_performance(client):
    """Recommendation logic must run in under 2 s (TMDB calls are mocked)."""
    empty = {"poster_url": None, "synopsis": None}
    with patch("api.main._tmdb_or_empty", new_callable=AsyncMock) as mock_tmdb:
        mock_tmdb.return_value = empty
        start = time.perf_counter()
        resp = await client.get("/recommend/movie/Toy%20Story")
        elapsed = time.perf_counter() - start

    assert resp.status_code == 200
    assert elapsed < 2.0, f"Recommendation took {elapsed:.2f}s — expected < 2s"


async def test_recommend_quality(client):
    """Every recommended film must appear in ratings_count.csv with count >= 50."""
    with open(_RATINGS_COUNT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        counts = {int(row["movieId"]): int(row["count"]) for row in reader}

    resp = await client.get("/recommend/movie/Toy%20Story")
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]

    assert len(recs) == 10

    for rec in recs:
        movie_id = rec["movieId"]
        assert movie_id in counts, f"movieId {movie_id} absent de ratings_count.csv"
        assert counts[movie_id] >= 50, (
            f"movieId {movie_id} a seulement {counts[movie_id]} ratings (< 50)"
        )
