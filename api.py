from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import redis
import json
import os

from .config import settings
from .embedding import embedder
from .ranker import SiameseRanker
from .explain import highlight_matches

app = FastAPI(title='NeuroMatch API')

r = redis.Redis.from_url(os.getenv('REDIS_URL', settings.REDIS_URL), decode_responses=False)

_ranker = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ranker():
    global _ranker
    if _ranker is None and os.path.exists(settings.RANKER_MODEL_PATH):
        model = SiameseRanker(emb_dim=embedder.model.get_sentence_embedding_dimension())
        model.load_state_dict(torch.load(settings.RANKER_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        _ranker = model

class MatchRequest(BaseModel):
    profile_text: str
    candidate_jobs: list

class MatchResponse(BaseModel):
    job_rankings: list

@app.on_event('startup')
async def startup_event():
    load_ranker()

@app.post('/match', response_model=MatchResponse)
async def match(req: MatchRequest):
    if not req.profile_text or not req.candidate_jobs:
        raise HTTPException(status_code=400, detail='profile_text and candidate_jobs required')

    key = f"match:{hash(req.profile_text)}:{len(req.candidate_jobs)}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    prof_emb = embedder.encode([req.profile_text])[0]
    job_embs = embedder.encode(req.candidate_jobs)

    prof_tensor = torch.tensor(np.repeat(prof_emb[None, :], len(job_embs), axis=0)).float().to(device)
    job_tensor = torch.tensor(job_embs).float().to(device)

    if _ranker is None:
        # fallback to cosine similarity
        scores = (job_embs @ prof_emb).tolist()
        order = np.argsort(-np.array(scores))
        probs = np.array(scores).tolist()
    else:
        with torch.no_grad():
            logits = _ranker(prof_tensor, job_tensor).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        order = np.argsort(-probs)

    results = []
    for idx in order:
        job_text = req.candidate_jobs[idx]
        score = float(probs[idx])
        prof_hl, job_hl = highlight_matches(req.profile_text, job_text, top_k_tokens=5)
        results.append({'job_text': job_text, 'score': score, 'profile_highlights': prof_hl, 'job_highlights': job_hl})

    out = {'job_rankings': results}
    r.set(key, json.dumps(out), ex=settings.CACHE_TTL)
    return out

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
