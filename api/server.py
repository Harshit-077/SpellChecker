"""
server.py - FastAPI Server for Hindi Spelling Correction
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import unicodedata

from model import Encoder, Decoder, Attention, Seq2Seq, correct_word # changed model_fixed to model here


# Config
VOCAB_PATH = "../data/vocab.txt"
MODEL_PATH = "../models/hindi_spelling_model.pt"
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_WORD_LEN = 50
MAX_INPUT_LEN = 256

# App
app = FastAPI(title="Hindi Spelling Corrector")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vocab
print("Loading vocabulary...")
vocab = {}
rev_vocab = {}

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        char = line.strip()
        vocab[char] = idx
        rev_vocab[idx] = char

PAD_IDX = vocab.get("<PAD>", 0)
SOS_IDX = vocab.get("<START>", vocab.get("<SOS>", 1))
EOS_IDX = vocab.get("<END>", vocab.get("<EOS>", 2))
UNK_IDX = vocab.get("<UNK>", 3)

print(f"✓ Loaded {len(vocab)} characters")
print(f"  PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")

# Load model
print("Loading model...")

attention = Attention(HIDDEN_SIZE)
encoder = Encoder(len(vocab), EMBED_SIZE, HIDDEN_SIZE, PAD_IDX).to(device)
decoder = Decoder(len(vocab), EMBED_SIZE, HIDDEN_SIZE, attention, PAD_IDX).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "encoder" in checkpoint:
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
else:
    model = Seq2Seq(encoder, decoder, device)
    model.load_state_dict(checkpoint)
    encoder = model.encoder
    decoder = model.decoder

encoder.eval()
decoder.eval()

print(f"✓ Model loaded")
print(f"  Device: {device}")
print(f"  Embed: {EMBED_SIZE}, Hidden: {HIDDEN_SIZE}")


# API
class SpellRequest(BaseModel):
    text: str


class SpellResponse(BaseModel):
    input: str
    corrected: str
    changed: bool


@app.get("/")
def root():
    return {
        "service": "Hindi Spelling Corrector",
        "model": "Seq2Seq + Attention",
        "vocab_size": len(vocab),
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE
    }


@app.post("/api/spell-check", response_model=SpellResponse)
def spell_check(req: SpellRequest):
    text = req.text.strip()
    
    if not text:
        return SpellResponse(input="", corrected="", changed=False)
    
    # Normalize
    text = unicodedata.normalize("NFC", text[:MAX_INPUT_LEN])
    
    # Correct each word
    words = text.split()
    corrected_words = []
    
    for word in words:
        try:
            corrected = correct_word(encoder, decoder, word, vocab, rev_vocab, device, MAX_WORD_LEN)
            corrected_words.append(corrected if corrected else word)
        except Exception as e:
            print(f"Error correcting '{word}': {e}")
            corrected_words.append(word)
    
    corrected = " ".join(corrected_words)
    
    return SpellResponse(
        input=text,
        corrected=corrected,
        changed=(text != corrected)
    )


@app.post("/api/correct-word")
def correct_single_word(req: SpellRequest):
    word = unicodedata.normalize("NFC", req.text.strip())
    
    if not word:
        return {"input": "", "corrected": "", "changed": False}
    
    try:
        corrected = correct_word(encoder, decoder, word, vocab, rev_vocab, device, MAX_WORD_LEN)
        return {
            "input": word,
            "corrected": corrected,
            "changed": (word != corrected)
        }
    except Exception as e:
        return {
            "input": word,
            "corrected": word,
            "changed": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("Starting Hindi Spelling Corrector API")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000)
