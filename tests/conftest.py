import pytest

import api.main as main_module
from api.config import MODELS_PATH
from api.recommender import Recommender


@pytest.fixture(scope="session", autouse=True)
def setup_recommender():
    """Load the ALS model once for the entire test session."""
    main_module.recommender = Recommender(MODELS_PATH)
