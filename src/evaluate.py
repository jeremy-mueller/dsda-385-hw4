import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MINDDataset, NRMSModel

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
NEG_SAMPLE_K = 4
MAX_HISTORY = 50
MAX_TITLE_LEN = 30
NUM_HEADS = 20
HEAD_DIM = 20
DROPOUT = 0.2


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1][:k]
    gains = np.array(y_true)[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_sorted = np.array(y_true)[order]
    for i, val in enumerate(y_sorted):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(model, val_loader, device):
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            history = batch["history"].to(device)
            candidates = batch["candidates"].to(device)
            labels = batch["labels"].numpy()
            hist_mask = batch["hist_mask"].to(device)

            scores = model(history, candidates, hist_mask).cpu().numpy()

            for i in range(len(labels)):
                y_true = labels[i]
                y_score = scores[i][: len(y_true)]
                if sum(y_true) == 0 or sum(y_true) == len(y_true):
                    continue
                aucs.append(roc_auc_score(y_true, y_score))
                mrrs.append(mrr_score(y_true, y_score))
                ndcg5s.append(ndcg_score(y_true, y_score, 5))
                ndcg10s.append(ndcg_score(y_true, y_score, 10))
    auc = np.mean(aucs)
    mrr = np.mean(mrrs)
    ndcg5 = np.mean(ndcg5s)
    ndcg10 = np.mean(ndcg10s)
    print(f"AUC: {auc:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"nDCG@5: {ndcg5:.4f}")
    print(f"nDCG@10: {ndcg10:.4f}")
    return {"AUC": auc, "MRR": mrr, "nDCG@5": ndcg5, "nDCG@10": ndcg10}


def extract_losses(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
    epoch_losses = []

    for file in files:
        path = os.path.join(folder_path, file)
        checkpoint = torch.load(path, map_location="cpu")

        if "epoch" in checkpoint and "loss" in checkpoint:
            epoch_losses.append((checkpoint["epoch"] + 1, checkpoint["loss"]))

    epoch_losses.sort(key=lambda x: x[0])
    return zip(*epoch_losses)


experiment_folders = {
    "Hyper 1": "../models/hyper1/",
    "Hyper 2": "../models/hyper2/",
    "Hyper 3": "../models/hyper3/",
}

plt.figure(figsize=(10, 6))

for label, folder in experiment_folders.items():
    if os.path.exists(folder):
        epochs, losses = extract_losses(folder)
        plt.plot(epochs, losses, marker="o", label=label)

plt.title("Training Loss Comparison Across Experiments")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.xticks(range(1, 6))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.savefig("../results/hyperparameter_comparison.png")

with open("../data/processed/MINDsmall_train.pkl", "rb") as f:
    data = pickle.load(f)

embedding_matrix = data["embedding_matrix"]

with open("../data/processed/MINDsmall_val.pkl", "rb") as f:
    data = pickle.load(f)

val_samples = data["train_samples"]

val_dataset = MINDDataset(val_samples)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NRMSModel(embedding_matrix, NUM_HEADS, HEAD_DIM, DROPOUT).to(device)

files = [f for f in os.listdir("../models/hyper3/") if f.endswith(".pt")]

best_loss = float("inf")
best_path = None

for file in files:
    path = os.path.join("../models/hyper3/", file)
    checkpoint = torch.load(path, map_location=device)

    current_loss = checkpoint.get("loss", float("inf"))
    if current_loss < best_loss:
        best_loss = current_loss
        best_path = path

if best_path:
    print(f"Evaluating {best_path} with training loss {best_loss}")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
else:
    print("No valid checkpoints found")

results = evaluate(model, val_loader, device)

results["model_path"] = best_path
results["batch_size"] = BATCH_SIZE

output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)

filename = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
save_path = os.path.join(output_dir, filename)

with open(save_path, "wb") as f:
    pickle.dump(results, f)

print(f"Evaluation results saved to {save_path}")
