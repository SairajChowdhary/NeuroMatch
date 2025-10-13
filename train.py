import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from .dataset import PairDataset
from .ranker import SiameseRanker
from .embedding import Embedder
from .config import settings

def train(csv_path, out_path=settings.RANKER_MODEL_PATH, epochs=3, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PairDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedder = Embedder()
    model = SiameseRanker(emb_dim=embedder.model.get_sentence_embedding_dimension()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0
        for profile_texts, job_texts, labels in pbar:
            with torch.no_grad():
                prof_emb = torch.tensor(embedder.encode(list(profile_texts))).float()
                job_emb = torch.tensor(embedder.encode(list(job_texts))).float()

            prof_emb = prof_emb.to(device)
            job_emb = job_emb.to(device)
            labels = labels.to(device)

            logits = model(prof_emb, job_emb)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        torch.save(model.state_dict(), out_path)

    print('Training completed. Model saved to', out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sample_data.csv')
    parser.add_argument('--out', type=str, default=settings.RANKER_MODEL_PATH)
    args = parser.parse_args()
    train(args.data, args.out)
