import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from news_encoder import NewsEncoder
from user_encoder import UserEncoder


class NRMSModel(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()
        news_dim = num_heads * head_dim
        self.news_encoder = NewsEncoder(embedding_matrix, num_heads, head_dim, dropout)
        self.user_encoder = UserEncoder(news_dim, num_heads, head_dim, dropout)

    def forward(self, history_ids, candidate_ids, hist_mask=None):
        batch, hist_len, tlen = history_ids.shape
        _, n_cand, _ = candidate_ids.shape

        hist_flat = history_ids.view(-1, tlen)
        hist_vecs = self.news_encoder(hist_flat)
        hist_vecs = hist_vecs.view(batch, hist_len, -1)

        cand_flat = candidate_ids.view(-1, tlen)
        cand_vecs = self.news_encoder(cand_flat)
        cand_vecs = cand_vecs.view(batch, n_cand, -1)

        user_vec = self.user_encoder(hist_vecs, hist_mask)

        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores


class MINDDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        hist_sum = s["history"].sum(axis=1)
        mask = (hist_sum > 0).astype(np.float32)

        return {
            "history": torch.tensor(s["history"], dtype=torch.long),
            "candidates": torch.tensor(s["candidates"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.float),
            "hist_mask": torch.tensor(mask, dtype=torch.float),
        }
