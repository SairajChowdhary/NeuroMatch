from src.embedding import embedder
from src.explain import highlight_matches
import numpy as np

def test_embedding_shape():
    texts = ["python developer", "data scientist"]
    emb = embedder.encode(texts)
    assert emb.shape[0] == 2
    assert emb.shape[1] > 0

def test_highlight_basic():
    profile = "Experienced python developer with ML"
    job = "Looking for python engineer with ML background"
    prof_hl, job_hl = highlight_matches(profile, job, top_k_tokens=3)
    assert len(prof_hl) > 0
    assert len(job_hl) > 0
