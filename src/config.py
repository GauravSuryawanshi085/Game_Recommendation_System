from pathlib import Path

# ==========================
# Project Root Directory
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================
# Data Paths
# ==========================
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "steam.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_games.csv"

# ==========================
# Model Paths
# ==========================
MODEL_DIR = BASE_DIR / "models"

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
SIMILARITY_MATRIX_PATH = MODEL_DIR / "similarity_matrix.pkl"
GAMES_DATA_PATH = MODEL_DIR / "games.pkl"

# ==========================
# Log Paths
# ==========================
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "recommendation_logs.csv"

# ==========================
# Recommendation Settings
# ==========================
DEFAULT_TOP_N = 5
MAX_FEATURES = 3000
RANDOM_STATE = 42