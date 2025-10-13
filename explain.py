from typing import List, Tuple
import numpy as np
from .embedding import embedder

def highlight_matches(profile: str, job: str, top_k_tokens=8) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    prof_tokens = profile.split()
    job_tokens = job.split()

    if len(prof_tokens) == 0 or len(job_tokens) == 0:
        return [], []

    prof_embs = embedder.encode(prof_tokens)
    job_embs = embedder.encode(job_tokens)

    sims = np.matmul(prof_embs, job_embs.T)
    prof_scores = sims.max(axis=1)
    job_scores = sims.max(axis=0)

    prof_top_idx = np.argsort(-prof_scores)[:top_k_tokens]
    job_top_idx = np.argsort(-job_scores)[:top_k_tokens]

    prof_highlights = [(prof_tokens[i], float(prof_scores[i])) for i in prof_top_idx]
    job_highlights = [(job_tokens[i], float(job_scores[i])) for i in job_top_idx]

    return prof_highlights, job_highlights
