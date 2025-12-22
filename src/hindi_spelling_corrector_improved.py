import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---

# CHANGE TO:
# BATCH_SIZE = 32          # Smaller (larger model needs memory)
# EMBED_DIM = 256          # 2x bigger embeddings
# HIDDEN_DIM = 512         # 2x bigger hidden layer  
# EPOCHS = 30              # Train longer
# LEARNING_RATE = 0.0005   # Slower learning (add this line)



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 512
EPOCHS = 30 # from 10 -> 20
MAX_LEN = 50  # Increased for longer words
LEARNING_RATE = 0.0005
CLIP_GRAD = 1.0  # Gradient clipping
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_PATH = 'hindi_spelling_model.pt'

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# --- 2. LOAD VOCABULARY ---
try:
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
except FileNotFoundError:
    print("Error: vocab.txt not found. Please create vocabulary file first.")
    exit(1)

char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = {i: char for i, char in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
PAD_IDX = char2idx['<PAD>']
SOS_IDX = char2idx['<START>']
EOS_IDX = char2idx['<END>']

print(f"Vocabulary size: {VOCAB_SIZE}")

# --- 3. DATASET & DATALOADER ---
class HindiSpellingDataset(Dataset):
    def __init__(self, data_df):
        self.df = data_df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def encode(self, text):
        """Convert string to list of indices with SOS and EOS tokens"""
        if not text or not isinstance(text, str):
            return torch.tensor([SOS_IDX, EOS_IDX])
        tokens = [SOS_IDX] + [char2idx.get(c, char2idx.get('<UNK>', 0)) for c in text] + [EOS_IDX]
        return torch.tensor(tokens)

    def __getitem__(self, idx):
        noisy = self.encode(str(self.df.iloc[idx, 0]))
        clean = self.encode(str(self.df.iloc[idx, 1]))
        return noisy, clean

def collate_fn(batch):
    """Pads sequences in the batch to the same length"""
    noisy_batch, clean_batch = zip(*batch)
    noisy_padded = pad_sequence(noisy_batch, batch_first=True, padding_value=PAD_IDX)
    clean_padded = pad_sequence(clean_batch, batch_first=True, padding_value=PAD_IDX)
    return noisy_padded.to(DEVICE), clean_padded.to(DEVICE)

# --- 4. MODEL ARCHITECTURE ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # Combine bidirectional hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))).unsqueeze(0)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        return attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden.squeeze(0), encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction.squeeze(1), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# --- 5. EVALUATION METRICS ---

def calculate_edit_distance(str1, str2):
    """Calculate Levenshtein distance between two strings"""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def calculate_cer(predictions, targets):
    """Calculate Character Error Rate"""
    total_distance = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        distance = calculate_edit_distance(pred, target)
        total_distance += distance
        total_chars += len(target)
    
    return total_distance / max(total_chars, 1)

def calculate_accuracy(predictions, targets):
    """Calculate exact match accuracy"""
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)

# --- 6. TRAINING & EVALUATION FUNCTIONS ---

def train_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    for src, trg in loader:
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        
        # Reshape for loss calculation
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for src, trg in loader:
            output = model(src, trg, 0)  # No teacher forcing during evaluation
            output_dim = output.shape[-1]
            
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            epoch_loss += loss.item()
            
            # Decode predictions for metrics
            for i in range(src.shape[0]):
                pred_indices = output[i].argmax(1).cpu().numpy()
                target_indices = trg[i].cpu().numpy()
                
                # Convert to strings (excluding SOS, EOS, PAD)
                pred_str = ''.join([idx2char[idx] for idx in pred_indices 
                                   if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])
                target_str = ''.join([idx2char[idx] for idx in target_indices 
                                     if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])
                
                predictions.append(pred_str)
                targets.append(target_str)
    
    avg_loss = epoch_loss / len(loader)
    cer = calculate_cer(predictions, targets)
    accuracy = calculate_accuracy(predictions, targets)
    
    return avg_loss, cer, accuracy

# --- 7. INFERENCE FUNCTION ---

def correct_spelling(model, word, max_len=MAX_LEN):
    """Correct spelling of a word"""
    if not word or not isinstance(word, str):
        return ""
    
    model.eval()
    with torch.no_grad():
        # Encode input
        word_encoded = [SOS_IDX] + [char2idx.get(c, char2idx.get('<UNK>', 0)) for c in word] + [EOS_IDX]
        src_tensor = torch.LongTensor(word_encoded).unsqueeze(0).to(DEVICE)
        
        # Encode
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Decode
        result = []
        input_token = torch.tensor([SOS_IDX]).to(DEVICE)
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            input_token = output.argmax(1)
            
            if input_token.item() == EOS_IDX:
                break
            if input_token.item() == PAD_IDX:
                continue
                
            result.append(idx2char.get(input_token.item(), ''))
        
        return "".join(result)

# --- 8. MAIN TRAINING LOOP ---

def main():
    # Load full dataset
    print("Loading dataset...")
    try:
        # Load full dataset (remove nrows to use all data)
        df = pd.read_csv('hindi_pairs.csv').dropna()  # Change to None for full dataset # changed from 200000 to 450000
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print("Error: hindi_pairs.csv not found")
        return
    
    # Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets and loaders
    train_dataset = HindiSpellingDataset(train_df)
    val_dataset = HindiSpellingDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    print("Initializing model...")
    enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    attn = Attention(HIDDEN_DIM)
    dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Training history
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_cers = []
    val_accs = []
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        # Decay teacher forcing ratio
        teacher_forcing_ratio = 0.5 * (0.95 ** epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP_GRAD, teacher_forcing_ratio)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_cer, val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_cers.append(val_cer)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | CER: {val_cer:.4f} | Acc: {val_acc:.4f}")
        print(f"  Teacher Forcing: {teacher_forcing_ratio:.3f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_cer': val_cer,
            'val_acc': val_acc,
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final CER: {val_cers[-1]:.4f}")
    print(f"Final Accuracy: {val_accs[-1]:.4f}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Test examples
    print("\n" + "=" * 80)
    print("Test Examples:")
    print("-" * 80)
    
    test_words = [
        'परिबर्तन',
        'भारत',
        'विद्यालय',
        'संस्कृति',
        'प्रधानमंत्री'
    ]
    
    for word in test_words:
        corrected = correct_spelling(model, word)
        print(f"Input: {word:20s} → Output: {corrected}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
