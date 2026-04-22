import os
import pickle
from collections import Counter

import nltk
import numpy as np
import pandas as pd

MAX_TITLE_LEN = 30
MAX_HIST_LEN = 50
NEG_K = 4
MIN_FREQ = 2

np.random.seed(21)


class NewsTokenizer:
    def __init__(self, max_title_len=MAX_TITLE_LEN, min_word_freq=MIN_FREQ):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, titles):
        word_counts = Counter()
        for title in titles:
            tokens = nltk.word_tokenize(title.lower())
            word_counts.update(tokens)
        for word, count in word_counts.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode_title(self, title):
        tokens = nltk.word_tokenize(title.lower())
        indices = [self.word2idx.get(t, 1) for t in tokens]
        if len(indices) < self.max_title_len:
            indices += [0] * (self.max_title_len - len(indices))
        else:
            indices = indices[: self.max_title_len]
        return indices


def load_glove(glove_path, word2idx, embed_dim=300):
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(
        "float32"
    )
    embedding_matrix[0] = 0
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                idx = word2idx[word]
                embedding_matrix[idx] = np.asarray(values[1:], dtype="float32")
                found += 1
    print(f"Found {found}/{vocab_size} words in GloVe")
    return embedding_matrix


def parse_behaviors(behaviors_df, news_encoded, neg_k=NEG_K, max_hist_len=MAX_HIST_LEN):
    samples = []
    padding_news = [0] * MAX_TITLE_LEN
    for _, row in behaviors_df.iterrows():
        raw_hist = row["history"].split() if pd.notna(row["history"]) else []
        hist_encoded = [news_encoded.get(nid, padding_news) for nid in raw_hist][
            -max_hist_len:
        ]
        if len(hist_encoded) < max_hist_len:
            hist_encoded = [padding_news] * (
                max_hist_len - len(hist_encoded)
            ) + hist_encoded
        impressions = row["impressions"].split()
        pos = [imp.split("-")[0] for imp in impressions if imp.endswith("-1")]
        neg = [imp.split("-")[0] for imp in impressions if imp.endswith("-0")]
        for p_id in pos:
            if p_id not in news_encoded:
                continue
            if len(neg) >= neg_k:
                sampled_neg = np.random.choice(neg, size=neg_k, replace=False)
            else:
                sampled_neg = (
                    np.random.choice(neg, size=neg_k, replace=True)
                    if len(neg) > 0
                    else []
                )
            candidate_ids = [p_id] + list(sampled_neg)
            candidate_encoded = [
                news_encoded.get(c, padding_news) for c in candidate_ids
            ]
            if len(candidate_encoded) == (neg_k + 1):
                samples.append(
                    {
                        "history": np.array(hist_encoded),
                        "candidates": np.array(candidate_encoded),
                        "labels": np.array([1] + [0] * neg_k),
                    }
                )
    return samples


news_cols = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
news_df = pd.read_csv("../data/MINDsmall_train/news.tsv", sep="\t", names=news_cols)

behavior_cols = ["impression_id", "user_id", "time", "history", "impressions"]
behaviors_df = pd.read_csv(
    "../data/MINDsmall_train/behaviors.tsv", sep="\t", names=behavior_cols
)

tokenizer = NewsTokenizer()
tokenizer.build_vocab(news_df["title"].tolist())

news_encoded = {
    row["news_id"]: tokenizer.encode_title(row["title"])
    for _, row in news_df.iterrows()
}

embedding_matrix = load_glove("../data/glove/glove.6B.300d.txt", tokenizer.word2idx)

print("Parsing behaviors")
train_samples = parse_behaviors(behaviors_df, news_encoded)
print(f"Generated {len(train_samples)} training samples.")

os.makedirs("../data/processed", exist_ok=True)

processed_data = {
    "train_samples": train_samples,
    "embedding_matrix": embedding_matrix,
    "word2idx": tokenizer.word2idx,
}

with open("../data/processed/MINDsmall_train.pkl", "wb") as f:
    pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Preprocessing complete. Data saved to ../data/processed/MINDsmall_train.pkl")

news_cols = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
news_df = pd.read_csv("../data/MINDsmall_dev/news.tsv", sep="\t", names=news_cols)

behavior_cols = ["impression_id", "user_id", "time", "history", "impressions"]
behaviors_df = pd.read_csv(
    "../data/MINDsmall_dev/behaviors.tsv", sep="\t", names=behavior_cols
)

tokenizer = NewsTokenizer()
tokenizer.build_vocab(news_df["title"].tolist())

news_encoded = {
    row["news_id"]: tokenizer.encode_title(row["title"])
    for _, row in news_df.iterrows()
}

embedding_matrix = load_glove("../data/glove/glove.6B.300d.txt", tokenizer.word2idx)

print("Parsing behaviors")
val_samples = parse_behaviors(behaviors_df, news_encoded)
print(f"Generated {len(val_samples)} validation samples.")

os.makedirs("../data/processed", exist_ok=True)

processed_data = {
    "val_samples": val_samples,
    "embedding_matrix": embedding_matrix,
    "word2idx": tokenizer.word2idx,
}

with open("../data/processed/MINDsmall_val.pkl", "wb") as f:
    pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Preprocessing complete. Data saved to ../data/processed/MINDsmall_val.pkl")
