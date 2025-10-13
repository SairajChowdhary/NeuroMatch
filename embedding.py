from sentence_transformers import SentenceTransformer
import numpy as np
from .config import settings

class Embedder:
    """Wrapper around a SentenceTransformer model for embedding text."""
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize: bool = True, batch_size: int = 32):
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        return emb

embedder = Embedder()
