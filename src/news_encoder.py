import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, X, mask=None):
        e = torch.tanh(self.proj(X))
        scores = self.query(e).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), X).squeeze(1)


class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()
        embed_dim = embedding_matrix.shape[1]
        self.word_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False
        )
        self.dropout = nn.Dropout(dropout)
        attn_dim = num_heads * head_dim
        self.proj = nn.Linear(embed_dim, attn_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )
        self.additive_attn = AdditiveAttention(attn_dim)

    def forward(self, title_ids):
        X = self.dropout(self.word_embed(title_ids))
        X = self.proj(X)
        X, _ = self.multihead_attn(X, X, X)
        X = self.dropout(X)
        news_vec = self.additive_attn(X)
        return news_vec
