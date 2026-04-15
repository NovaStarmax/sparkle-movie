from pydantic import BaseModel
from typing import Optional


class MovieInfo(BaseModel):
    movieId: int
    title: str
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    synopsis: Optional[str] = None


class RecommendedMovie(BaseModel):
    movieId: int
    title: str
    genres: list[str]
    predicted_score: float
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    synopsis: Optional[str] = None
    release_year: Optional[int] = None


class RecommendResponse(BaseModel):
    query: str
    matched_movie: MovieInfo
    recommendations: list[RecommendedMovie]


class SearchResult(BaseModel):
    movieId: int
    title: str


class HealthResponse(BaseModel):
    status: str
    model: str
    n_users: int
    n_items: int
    rmse: float
