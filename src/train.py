import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MINDDataset, NRMSModel

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
NEG_SAMPLE_K = 4
MAX_HISTORY = 50
DROPOUT = 0.2
MAX_TITLE_LEN = 30
NUM_HEADS = 20
HEAD_DIM = 20

torch.manual_seed(21)


with open("../data/processed/MINDsmall_train.pkl", "rb") as f:
    data = pickle.load(f)

train_samples = data["train_samples"]
embedding_matrix = data["embedding_matrix"]

train_dataset = MINDDataset(train_samples)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

model = NRMSModel(embedding_matrix, NUM_HEADS, HEAD_DIM, DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.CrossEntropyLoss()

os.makedirs("../models", exist_ok=True)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader):
        history = batch["history"].to(device)
        candidates = batch["candidates"].to(device)
        labels = batch["labels"].to(device)
        hist_mask = batch["hist_mask"].to(device)

        optimizer.zero_grad()
        scores = model(history, candidates, hist_mask)

        target = torch.zeros(scores.size(0), dtype=torch.long).to(device)
        loss = criterion(scores, target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    current_lr = optimizer.param_groups[0]["lr"]
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Current LR: {current_lr}")

    scheduler.step()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        f"../models/nrms_epoch{epoch + 1}.pt",
    )
