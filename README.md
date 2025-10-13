NeuroMatch
==========

Context-aware job matching system built for production use. This repository contains the core components needed to train, index, retrieve, rank, and serve job recommendations at scale.

Structure
- src/: core application code
- data/: sample dataset for quick runs
- infra/: Kubernetes manifests and Redis config
- .github/: CI workflow
- tests/: unit tests and small integration checks
- docker-compose.yml, Dockerfile, requirements.txt

Quick start (development)
1. Install dependencies: pip install -r requirements.txt
2. Start services: docker-compose up --build
3. Train a small model or use provided sample model
4. Run API: POST /match with profile_text and candidate_jobs

See src/ for implementation details.
