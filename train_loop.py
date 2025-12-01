# train_loop.py

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from py_dataset import HindiSpellDataset, collate_fn, vocab, rev_vocab
from word_to_seq import Encoder, Decoder, Seq2Seq, device

# -----------------------------
# Hyperparameters
# -----------------------------
embed_size = 256
hidden_size = 512
num_layers = 2
dropout = 0.1
batch_size = 64
num_epochs = 5
teacher_forcing_ratio = 0.5
lr = 0.001

PAD_IDX = vocab["<PAD>"]

# -----------------------------
# Load dataset
# -----------------------------
with open("data/data_pairs.pkl", "rb") as f:
    data_pairs = pickle.load(f)
print(f"Loaded {len(data_pairs)} sentence pairs")

dataset = HindiSpellDataset(data_pairs, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# -----------------------------
# Initialize model
# -----------------------------
vocab_size = len(vocab)
encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Training loop
# -----------------------------
model.train()
for epoch in range(1, num_epochs+1):
    epoch_loss = 0
    for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, src_lengths, tgt, teacher_forcing_ratio)

        # output: [batch, tgt_len, vocab_size]
        # Shift tgt to match predictions (exclude <SOS>)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} Average Loss: {epoch_loss / len(dataloader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/seq2seq_epoch{epoch}.pt")
    print(f"Checkpoint saved for epoch {epoch}")
