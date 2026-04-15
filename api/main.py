import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import MODELS_PATH
from api.models import (
    HealthResponse,
    MovieInfo,
    RecommendedMovie,
    RecommendResponse,
    SearchResult,
)
from api.recommender import Recommender
from api.tmdb import fetch_movie_details

# Global recommender — loaded once at startup
recommender: Optional[Recommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = Recommender(MODELS_PATH)
    yield
    # Nothing to clean up for a read-only model


app = FastAPI(
    title="Sparkle Movie API",
    description="ALS-based movie recommendation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _tmdb_or_empty(
    client: httpx.AsyncClient, tmdb_id: Optional[int]
) -> dict:
    if tmdb_id is None:
        return {"poster_url": None, "synopsis": None}
    return await fetch_movie_details(client, tmdb_id)


def _extract_year(title: str) -> Optional[int]:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    meta = recommender.metadata
    return HealthResponse(
        status="ok",
        model=meta.get("model", "ALS"),
        n_users=meta.get("n_users", 0),
        n_items=meta.get("n_items", 0),
        rmse=meta.get("rmse", 0.0),
    )


@app.get("/search", response_model=list[SearchResult])
async def search(q: str = Query(..., min_length=1)) -> list[SearchResult]:
    results = recommender.search(q, n=5)
    return [SearchResult(**r) for r in results]


@app.get("/recommend/movie/{movie_title}", response_model=RecommendResponse)
async def recommend_movie(movie_title: str) -> RecommendResponse:
    movie = recommender.find_movie(movie_title)
    if movie is None:
        raise HTTPException(
            status_code=404, detail=f"Movie '{movie_title}' not found"
        )

    movie_id = int(movie["movieId"])
    recs = recommender.recommend(movie_id, n=10, top_users=50)

    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for '{movie_title}'",
        )

    matched_tmdb_id = recommender.get_tmdb_id(movie_id)
    rec_tmdb_ids = [recommender.get_tmdb_id(r["movieId"]) for r in recs]

    # Fetch all TMDB details concurrently (gracefully degrades on failure)
    async with httpx.AsyncClient() as client:
        all_details = await asyncio.gather(
            _tmdb_or_empty(client, matched_tmdb_id),
            *[_tmdb_or_empty(client, tid) for tid in rec_tmdb_ids],
        )

    matched_details = all_details[0]
    rec_details = all_details[1:]

    matched_movie = MovieInfo(
        movieId=movie_id,
        title=movie["title"],
        tmdb_id=matched_tmdb_id,
        poster_url=matched_details["poster_url"],
        synopsis=matched_details["synopsis"],
    )

    recommendations = [
        RecommendedMovie(
            movieId=r["movieId"],
            title=r["title"],
            genres=r["genres"],
            predicted_score=round(r["predicted_score"], 4),
            tmdb_id=tid,
            poster_url=details["poster_url"],
            synopsis=details["synopsis"],
            release_year=_extract_year(r["title"]),
        )
        for r, details, tid in zip(recs, rec_details, rec_tmdb_ids)
    ]

    return RecommendResponse(
        query=movie_title,
        matched_movie=matched_movie,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Static frontend — mounted LAST so API routes take priority
# ---------------------------------------------------------------------------

_frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
