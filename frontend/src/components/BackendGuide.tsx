import { useState } from "react";
import { Copy, Check, Server, Code, Zap, Link, Terminal, FileCode, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "./ui/button";
import { toast } from "sonner";

const CodeBlock = ({ code, language, filename }: { code: string; language: string; filename?: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    toast.success("Code copied!");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative rounded-xl overflow-hidden border border-border bg-foreground/5">
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 bg-secondary/50 border-b border-border">
          <div className="flex items-center gap-2">
            <FileCode className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-foreground">{filename}</span>
          </div>
          <span className="text-xs text-muted-foreground">{language}</span>
        </div>
      )}
      <div className="relative">
        <pre className="p-4 overflow-x-auto text-sm">
          <code className="text-foreground/90 font-mono whitespace-pre">{code}</code>
        </pre>
        <button
          onClick={handleCopy}
          className="absolute top-3 right-3 p-2 bg-secondary hover:bg-secondary/80 rounded-lg transition-all duration-200"
          title="Copy code"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4 text-muted-foreground" />
          )}
        </button>
      </div>
    </div>
  );
};

const Section = ({ 
  title, 
  icon: Icon, 
  children, 
  defaultOpen = false 
}: { 
  title: string; 
  icon: React.ElementType; 
  children: React.ReactNode;
  defaultOpen?: boolean;
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-border rounded-2xl overflow-hidden bg-card shadow-card">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 md:p-6 hover:bg-secondary/30 transition-colors duration-200"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl gradient-saffron flex items-center justify-center">
            <Icon className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold text-foreground">{title}</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-5 h-5 text-muted-foreground" />
        )}
      </button>
      {isOpen && (
        <div className="px-4 md:px-6 pb-6 space-y-4 animate-fade-in">
          {children}
        </div>
      )}
    </div>
  );
};

const BackendGuide = () => {
  const requirementsCode = `torch>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pydantic>=2.5.0`;

  const modelCode = `# model.py - Model definitions (same as training)
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=0.0, max_len=200):
        batch_size = src.size(0)
        vocab_size = self.decoder.fc.out_features
        
        encoder_out, hidden = self.encoder(src, src_lengths)
        
        # Start with SOS token (index 1)
        input_token = torch.tensor([1] * batch_size, device=self.device)
        
        outputs = []
        for t in range(max_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            
            # Greedy decoding
            input_token = output.argmax(1)
            
            # Stop if all sequences have generated EOS (index 2)
            if (input_token == 2).all():
                break
        
        return torch.stack(outputs, dim=1)`;

  const serverCode = `# server.py - FastAPI backend
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import Encoder, Decoder, Seq2Seq

app = FastAPI(title="Hindi Spell Checker API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and vocab
model = None
vocab = None
rev_vocab = None
device = None


class SpellCheckRequest(BaseModel):
    text: str


class SpellCheckResponse(BaseModel):
    original: str
    corrected: str
    success: bool


def load_model():
    """Load the trained model and vocabulary."""
    global model, vocab, rev_vocab, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(
        "checkpoints/seq2seq_best.pt",
        map_location=device,
        weights_only=False
    )
    
    vocab = checkpoint["vocab"]
    rev_vocab = {idx: char for char, idx in vocab.items()}
    vocab_size = len(vocab)
    
    # Model hyperparameters (must match training)
    embed_size = 192
    hidden_size = 256
    num_layers = 2
    
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print("Model loaded successfully!")


def preprocess_text(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert text to tensor."""
    # Character-level tokenization
    UNK_IDX = vocab.get("<UNK>", 3)
    EOS_IDX = vocab.get("<EOS>", 2)
    
    ids = [vocab.get(c, UNK_IDX) for c in text] + [EOS_IDX]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    
    return src, lengths


def decode_output(output_ids: list[int]) -> str:
    """Convert output IDs back to text."""
    ignore = {vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"]}
    chars = [rev_vocab.get(i, "") for i in output_ids if i not in ignore]
    return "".join(chars)


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/api/spell-check", response_model=SpellCheckResponse)
async def spell_check(request: SpellCheckRequest):
    """Check and correct spelling in Hindi text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Preprocess
        src, lengths = preprocess_text(request.text)
        
        # Inference
        with torch.no_grad():
            output = model(src, lengths)
            pred_ids = output.argmax(dim=-1)[0].cpu().tolist()
        
        # Decode
        corrected = decode_output(pred_ids)
        
        return SpellCheckResponse(
            original=request.text,
            corrected=corrected,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`;

  const frontendCode = `// Update handleCheckSpelling in SpellChecker.tsx
const handleCheckSpelling = async () => {
  if (!inputText.trim()) {
    toast.error("कृपया कुछ टेक्स्ट दर्ज करें");
    return;
  }

  setIsProcessing(true);
  setHasChecked(false);

  try {
    const response = await fetch("http://localhost:8000/api/spell-check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: inputText }),
    });

    if (!response.ok) {
      throw new Error("Spell check failed");
    }

    const data = await response.json();
    setOutputText(data.corrected);
    setHasChecked(true);
    toast.success("वर्तनी जाँच पूर्ण!");
  } catch (error) {
    console.error("Error:", error);
    toast.error("सर्वर से कनेक्ट नहीं हो सका");
  } finally {
    setIsProcessing(false);
  }
};`;

  const runCommands = `# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py

# Or with uvicorn directly
uvicorn server:app --reload --host 0.0.0.0 --port 8000`;

  const folderStructure = `backend/
├── checkpoints/
│   └── seq2seq_best.pt    # Your trained model
├── model.py               # Model definitions
├── server.py              # FastAPI server
└── requirements.txt       # Python dependencies`;

  return (
    <div className="w-full max-w-6xl mx-auto px-4 md:px-8 py-12">
      <div className="text-center mb-10 animate-fade-in">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent/10 rounded-full mb-4">
          <Server className="w-4 h-4 text-accent" />
          <span className="text-sm font-medium text-accent">Backend Integration Guide</span>
        </div>
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-3">
          Connect Your Seq2Seq Model
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Follow this guide to set up a FastAPI backend that loads your trained 
          <code className="mx-1 px-2 py-0.5 bg-secondary rounded text-sm">seq2seq_best.pt</code> 
          model and exposes an API for spell checking.
        </p>
      </div>

      <div className="space-y-4">
        <Section title="1. Project Structure" icon={Terminal} defaultOpen={true}>
          <p className="text-sm text-muted-foreground mb-4">
            Create a backend folder with the following structure:
          </p>
          <CodeBlock code={folderStructure} language="plaintext" filename="Folder Structure" />
        </Section>

        <Section title="2. Requirements" icon={Code}>
          <p className="text-sm text-muted-foreground mb-4">
            Create a <code className="px-1.5 py-0.5 bg-secondary rounded text-xs">requirements.txt</code> file:
          </p>
          <CodeBlock code={requirementsCode} language="txt" filename="requirements.txt" />
        </Section>

        <Section title="3. Model Definition" icon={Zap}>
          <p className="text-sm text-muted-foreground mb-4">
            Copy your model architecture to <code className="px-1.5 py-0.5 bg-secondary rounded text-xs">model.py</code>. 
            This must match your training code exactly:
          </p>
          <CodeBlock code={modelCode} language="python" filename="model.py" />
        </Section>

        <Section title="4. FastAPI Server" icon={Server}>
          <p className="text-sm text-muted-foreground mb-4">
            Create the API server that loads the model and handles requests:
          </p>
          <CodeBlock code={serverCode} language="python" filename="server.py" />
        </Section>

        <Section title="5. Frontend Integration" icon={Link}>
          <p className="text-sm text-muted-foreground mb-4">
            Update the <code className="px-1.5 py-0.5 bg-secondary rounded text-xs">handleCheckSpelling</code> function 
            in your React component to call the API:
          </p>
          <CodeBlock code={frontendCode} language="typescript" filename="SpellChecker.tsx" />
        </Section>

        <Section title="6. Run the Server" icon={Terminal}>
          <p className="text-sm text-muted-foreground mb-4">
            Start the backend server:
          </p>
          <CodeBlock code={runCommands} language="bash" filename="Terminal" />
          <div className="mt-4 p-4 bg-primary/5 border border-primary/20 rounded-xl">
            <p className="text-sm text-foreground">
              <strong>Note:</strong> The server runs on <code className="px-1.5 py-0.5 bg-secondary rounded text-xs">http://localhost:8000</code>. 
              Make sure CORS is configured if deploying to different domains.
            </p>
          </div>
        </Section>
      </div>

      <div className="mt-10 p-6 bg-card rounded-2xl border border-border shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
            <Zap className="w-4 h-4 text-accent" />
          </div>
          API Endpoints
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-secondary/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-1 bg-green-500/20 text-green-600 text-xs font-bold rounded">POST</span>
              <code className="text-sm font-mono text-foreground">/api/spell-check</code>
            </div>
            <p className="text-sm text-muted-foreground">
              Submit Hindi text for spell correction. Returns original and corrected text.
            </p>
          </div>
          <div className="p-4 bg-secondary/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-1 bg-blue-500/20 text-blue-600 text-xs font-bold rounded">GET</span>
              <code className="text-sm font-mono text-foreground">/api/health</code>
            </div>
            <p className="text-sm text-muted-foreground">
              Health check endpoint to verify server and model status.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BackendGuide;
