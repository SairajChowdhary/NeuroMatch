import numpy as np
from .faiss_index import FaissIndex
from .config import settings

class Retriever:
    def __init__(self, dim, index_path=settings.FAISS_INDEX_PATH, n_list=100):
        self.index = FaissIndex(dim, index_path=index_path, n_list=n_list)
        if index_path and os.path.exists(index_path):
            try:
                self.index.load()
            except Exception:
                pass

    def build(self, embeddings):
        self.index.build_ivf(embeddings)

    def query(self, query_emb, top_k=10):
        D, I = self.index.search(query_emb, top_k)
        return D, I
