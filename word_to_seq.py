import torch
import torch.nn as nn
import torch.nn.functional as F

# Load vocab from file
vocab = {}
with open("vocab/hindi_vocab_100k.tsv", "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split("\t")
        vocab[word] = int(idx)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        """
        x: [batch_size, seq_len]
        lengths: [batch_size] (actual lengths of sequences)
        """
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_size]
        
        # Pack sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h, c)  # out: [batch, seq_len, hidden], h/c: [num_layers, batch, hidden]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        """
        x: [batch_size] current word indices
        hidden: (h, c) from previous step
        """
        x = x.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(x))  # [batch, 1, embed_size]
        output, hidden = self.lstm(embedded, hidden)  # output: [batch, 1, hidden]
        output = self.fc(output.squeeze(1))  # [batch, vocab_size]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        src_lengths: [batch]
        tgt: [batch, tgt_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        # Encoder
        encoder_out, hidden = self.encoder(src, src_lengths)
        
        # First input to decoder = <SOS>
        input = tgt[:, 0]  # [batch]
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab)
embed_size = 256
hidden_size = 512
num_layers = 2
dropout = 0.1

encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)

# Training setup

import torch.optim as optim

PAD_IDX = vocab["<PAD>"]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

