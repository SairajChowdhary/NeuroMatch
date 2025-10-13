import torch
import torch.nn as nn

class SiameseRanker(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_a, emb_b):
        x = torch.cat([emb_a, emb_b], dim=1)
        out = self.mlp(x)
        return out.squeeze(1)
