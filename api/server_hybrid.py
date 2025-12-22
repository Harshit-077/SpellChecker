"""
Hybrid Hindi Spell Checker API Server
======================================
FastAPI server combining Edit-Distance + Neural correction
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import unicodedata
from typing import List, Optional

from hybrid_spell_checker import HybridSpellChecker

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Hybrid Hindi Spell Checker API",
    description="Combines Edit-Distance and Neural correction for optimal results",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global checker instance
checker = None
DEVICE = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SpellCheckRequest(BaseModel):
    text: str
    strategy: Optional[str] = "hybrid"  # "hybrid", "neural", or "edit-distance"

class WordCorrection(BaseModel):
    original: str
    corrected: str
    changed: bool
    edit_distance_suggestion: Optional[str] = None
    neural_suggestion: Optional[str] = None
    method_used: str
    confidence: float

class SpellCheckResponse(BaseModel):
    input: str
    corrected: str
    changed: bool
    words: List[WordCorrection]
    strategy_used: str

class HealthResponse(BaseModel):
    status: str
    components: dict

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the hybrid spell checker on startup"""
    global checker, DEVICE
    
    print("=" * 80)
    print("STARTING HYBRID HINDI SPELL CHECKER API")
    print("=" * 80)
    
    # Detect device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    print(f"\nDevice: {DEVICE}")
    
    # Initialize hybrid checker
    try:
        checker = HybridSpellChecker(
            model_path='hindi_spelling_model_split.pt',
            vocab_path='vocab.txt',
            dictionary_path='dictionary_clean.txt',
            device=DEVICE
        )
        print("\n" + "=" * 80)
        print("API SERVER READY")
        print("=" * 80)
        print("\nEndpoints:")
        print("  GET  /              - API info")
        print("  GET  /health        - Health check")
        print("  POST /api/spell-check - Spell check text")
        print("  POST /api/correct-word - Correct single word")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error initializing checker: {e}")
        print("Server will start but spell checking will not work!")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """API information"""
    return {
        "name": "Hybrid Hindi Spell Checker API",
        "version": "2.0.0",
        "description": "Combines Edit-Distance (Levenshtein) and Neural (Seq2Seq) correction",
        "components": {
            "edit_distance": {
                "algorithm": "Levenshtein",
                "dictionary_size": "90,451 words",
                "accuracy": "~55%"
            },
            "neural": {
                "architecture": "Seq2Seq + Attention",
                "parameters": "8.3M",
                "accuracy": "~68%"
            },
            "hybrid": {
                "strategy": "Intelligent fusion",
                "accuracy": "~75-80%"
            }
        },
        "endpoints": {
            "spell_check": "/api/spell-check",
            "correct_word": "/api/correct-word",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    if checker is None:
        return HealthResponse(
            status="error",
            components={
                "api": "running",
                "checker": "not initialized"
            }
        )
    
    return HealthResponse(
        status="healthy",
        components={
            "api": "running",
            "edit_distance": "ready",
            "neural_model": "ready",
            "device": DEVICE
        }
    )


@app.post("/api/spell-check", response_model=SpellCheckResponse)
def spell_check(request: SpellCheckRequest):
    """
    Spell check and correct Hindi text
    
    Strategies:
    - "hybrid" (default): Best of both methods
    - "neural": Neural model only
    - "edit-distance": Dictionary-based only
    """
    if checker is None:
        return SpellCheckResponse(
            input=request.text,
            corrected=request.text,
            changed=False,
            words=[],
            strategy_used="error"
        )
    
    # Normalize input
    text = unicodedata.normalize('NFC', request.text.strip())
    
    if not text:
        return SpellCheckResponse(
            input="",
            corrected="",
            changed=False,
            words=[],
            strategy_used=request.strategy
        )
    
    # Split into words
    words = text.split()
    corrected_words = []
    word_details = []
    
    for word in words:
        try:
            # Get correction with all details
            result = checker.correct(word, strategy=request.strategy)
            
            corrected_word = result['final_correction']
            corrected_words.append(corrected_word)
            
            # Add detailed info for each word
            word_details.append(WordCorrection(
                original=word,
                corrected=corrected_word,
                changed=(word != corrected_word),
                edit_distance_suggestion=result['edit_distance_correction'],
                neural_suggestion=result['neural_correction'],
                method_used=result['method_used'],
                confidence=result['confidence']
            ))
            
        except Exception as e:
            print(f"Error correcting '{word}': {e}")
            corrected_words.append(word)
            word_details.append(WordCorrection(
                original=word,
                corrected=word,
                changed=False,
                method_used="error",
                confidence=0.0
            ))
    
    # Join corrected text
    corrected_text = " ".join(corrected_words)
    
    return SpellCheckResponse(
        input=text,
        corrected=corrected_text,
        changed=(text != corrected_text),
        words=word_details,
        strategy_used=request.strategy
    )


@app.post("/api/correct-word")
def correct_word(request: SpellCheckRequest):
    """
    Correct a single word with detailed information
    
    Returns both edit-distance and neural suggestions
    plus the final hybrid decision
    """
    if checker is None:
        return {
            "error": "Spell checker not initialized",
            "input": request.text,
            "corrected": request.text
        }
    
    word = unicodedata.normalize('NFC', request.text.strip())
    
    if not word:
        return {
            "input": "",
            "corrected": "",
            "changed": False
        }
    
    try:
        result = checker.correct(word, strategy=request.strategy or "hybrid")
        
        return {
            "input": result['input'],
            "corrected": result['final_correction'],
            "changed": (result['input'] != result['final_correction']),
            "details": {
                "edit_distance_suggestion": result['edit_distance_correction'],
                "neural_suggestion": result['neural_correction'],
                "method_used": result['method_used'],
                "confidence": result['confidence']
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "input": word,
            "corrected": word,
            "changed": False
        }


@app.post("/api/compare-methods")
def compare_methods(request: SpellCheckRequest):
    """
    Compare all three correction methods side-by-side
    
    Useful for debugging and demonstration
    """
    if checker is None:
        return {"error": "Spell checker not initialized"}
    
    word = unicodedata.normalize('NFC', request.text.strip())
    
    if not word:
        return {"error": "Empty input"}
    
    try:
        comparison = checker.compare_methods(word)
        
        return {
            "input": word,
            "results": {
                "edit_distance": comparison['edit_distance'],
                "neural": comparison['neural'],
                "hybrid": comparison['hybrid']
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("Starting Hybrid Hindi Spell Checker API Server")
    print("=" * 80)
    print("\nAccess at:")
    print("  - API: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
