"""
model.py - Hindi Spelling Correction Model  # also changed the name of this file from model_fixed.py to model.py along with the same modification in the server.py
Architecture: Bidirectional LSTM + Attention (256/512)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Bidirectional LSTM encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.rnn(embedded)
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))).unsqueeze(0)
        
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau attention"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        return attention


class Decoder(nn.Module):
    """LSTM decoder with attention"""
    def __init__(self, vocab_size, embed_size, hidden_size, attention, pad_idx, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(hidden_size * 2 + embed_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 3 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        a = self.attention(hidden.squeeze(0), encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction.squeeze(1), hidden, cell


class Seq2Seq(nn.Module):
    """Complete model"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        import random
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        
        return outputs


def correct_word(encoder, decoder, word, vocab, rev_vocab, device, max_len=50):
    """Correct a single word"""
    encoder.eval()
    decoder.eval()
    
    # Get token indices (handle both naming conventions)
    PAD_IDX = vocab.get("<PAD>", 0)
    SOS_IDX = vocab.get("<START>", vocab.get("<SOS>", 1))
    EOS_IDX = vocab.get("<END>", vocab.get("<EOS>", 2))
    UNK_IDX = vocab.get("<UNK>", 3)
    
    with torch.no_grad():
        # Encode word
        src_ids = [SOS_IDX] + [vocab.get(c, UNK_IDX) for c in word] + [EOS_IDX]
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # Encode
        encoder_outputs, hidden, cell = encoder(src)
        
        # Decode
        input_token = torch.tensor([SOS_IDX], device=device)
        decoded_ids = []
        
        for _ in range(max_len):
            output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
            pred = output.argmax(dim=1).item()
            
            if pred == EOS_IDX:
                break
            
            decoded_ids.append(pred)
            input_token = torch.tensor([pred], device=device)
    
    # Convert to string
    chars = [rev_vocab.get(i, "") for i in decoded_ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX)]
    return "".join(chars)
