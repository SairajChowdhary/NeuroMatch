import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def synthetic_texts(n):
    return ["software engineer with python and ml skills " + str(i) for i in range(n)]

def build_embeddings(model_name='sentence-transformers/all-mpnet-base-v2', n=50000, batch=1024):
    model = SentenceTransformer(model_name)
    texts = synthetic_texts(n)
    emb = model.encode(texts, batch_size=batch, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    return emb.astype('float32')

def run_demo():
    emb = build_embeddings(n=10000)
    dim = emb.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    nlist = 128
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    q = emb[:256]
    k = 10
    t0 = time.time()
    D, I = index.search(q, k)
    t1 = time.time()
    print('Avg latency ms:', (t1-t0)/q.shape[0]*1000)

if __name__ == '__main__':
    run_demo()
