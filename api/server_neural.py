"""
Neural-Only Hindi Spell Checker API
====================================
Pure neural seq2seq correction (no edit-distance)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import unicodedata
from typing import List

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))).unsqueeze(0)
        return outputs, hidden, cell


class Attention(nn.Module):
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

# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(
    title="Neural Hindi Spell Checker API",
    description="Pure neural seq2seq correction with attention",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
encoder = None
decoder = None
vocab = {}
rev_vocab = {}
device = None
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SpellCheckRequest(BaseModel):
    text: str

class WordCorrection(BaseModel):
    original: str
    corrected: str
    changed: bool

class SpellCheckResponse(BaseModel):
    input: str
    corrected: str
    changed: bool
    words: List[WordCorrection]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def correct_word(word: str, max_len: int = 50) -> str:
    """Correct a single word using neural model"""
    word = unicodedata.normalize('NFC', word)
    
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
    chars = [
        rev_vocab.get(i, "") 
        for i in decoded_ids 
        if i not in (PAD_IDX, SOS_IDX, EOS_IDX)
    ]
    
    return "".join(chars)

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global encoder, decoder, vocab, rev_vocab, device
    global PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
    
    print("=" * 80)
    print("STARTING NEURAL HINDI SPELL CHECKER API")
    print("=" * 80)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"\nDevice: {device}")
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            vocab[char] = idx
            rev_vocab[idx] = char
    
    PAD_IDX = vocab.get("<PAD>", 0)
    SOS_IDX = vocab.get("<START>", vocab.get("<SOS>", 1))
    EOS_IDX = vocab.get("<END>", vocab.get("<EOS>", 2))
    UNK_IDX = vocab.get("<UNK>", 3)
    
    print(f"✓ Loaded vocabulary: {len(vocab)} characters")
    
    # Load model
    print("\nLoading neural model...")
    attention = Attention(512)
    encoder = Encoder(len(vocab), 256, 512, PAD_IDX).to(device)
    decoder = Decoder(len(vocab), 256, 512, attention, PAD_IDX).to(device)
    
    checkpoint = torch.load('hindi_spelling_model_split.pt', map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    print("✓ Neural model loaded")
    print(f"  Parameters: 8.3M")
    print(f"  Architecture: Seq2Seq + Attention")
    
    print("\n" + "=" * 80)
    print("NEURAL API READY - Running on port 8001")
    print("=" * 80)
    print("\nEndpoints:")
    print("  GET  / - API info")
    print("  POST /api/spell-check - Correct text")
    print("=" * 80 + "\n")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "name": "Neural Hindi Spell Checker API",
        "version": "1.0.0",
        "method": "Seq2Seq with Attention",
        "parameters": "8.3M",
        "accuracy": "~68%",
        "port": 8001
    }

@app.post("/api/spell-check", response_model=SpellCheckResponse)
def spell_check(request: SpellCheckRequest):
    text = unicodedata.normalize('NFC', request.text.strip())
    
    if not text:
        return SpellCheckResponse(
            input="", corrected="", changed=False, words=[]
        )
    
    words = text.split()
    corrected_words = []
    word_details = []
    
    for word in words:
        try:
            corrected = correct_word(word)
            corrected_words.append(corrected)
            word_details.append(WordCorrection(
                original=word,
                corrected=corrected,
                changed=(word != corrected)
            ))
        except Exception as e:
            print(f"Error: {e}")
            corrected_words.append(word)
            word_details.append(WordCorrection(
                original=word, corrected=word, changed=False
            ))
    
    corrected_text = " ".join(corrected_words)
    
    return SpellCheckResponse(
        input=text,
        corrected=corrected_text,
        changed=(text != corrected_text),
        words=word_details
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
