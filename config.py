from pydantic import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RANKER_MODEL_PATH: str = "models/ranker.pt"
    TOP_K: int = 10
    CACHE_TTL: int = 3600
    FAISS_INDEX_PATH: str = "models/faiss.index"

settings = Settings()
