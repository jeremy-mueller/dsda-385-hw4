import torch.nn as nn

from news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=news_dim, num_heads=num_heads, batch_first=True
        )
        self.additive_attn = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clicked_news_vecs, mask=None):
        X, _ = self.multihead_attn(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        X = self.dropout(X)
        user_vec = self.additive_attn(X, mask)
        return user_vec
