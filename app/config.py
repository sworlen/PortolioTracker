import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///portfolio.db")
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "30"))
REFRESH_TOKEN_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "14"))
