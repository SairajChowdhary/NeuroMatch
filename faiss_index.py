import os
import numpy as np
import faiss

class FaissIndex:
    def __init__(self, dim, index_path=None, n_list=100):
        self.dim = dim
        self.index_path = index_path
        self.n_list = n_list
        self.index = None

    def build_ivf(self, embeddings: np.ndarray):
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_list, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
        self.index = index
        if self.index_path:
            faiss.write_index(index, self.index_path)

    def load(self):
        if self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError('Index file not found')

    def search(self, query_emb: np.ndarray, top_k=10):
        if self.index is None:
            raise RuntimeError('Index not initialized')
        D, I = self.index.search(query_emb, top_k)
        return D, I
