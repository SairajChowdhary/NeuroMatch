import pandas as pd
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.pairs = df.to_dict('records')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        return rec['profile_text'], rec['job_text'], float(rec['label'])
