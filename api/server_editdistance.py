"""
Edit-Distance Hindi Spell Checker API
======================================
Pure Levenshtein algorithm (no neural model)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import unicodedata
from typing import List, Optional

# ============================================================================
# LEVENSHTEIN ALGORITHM
# ============================================================================

class LevenshteinSpellChecker:
    """Traditional edit-distance spell checker"""
    
    def __init__(self, dictionary_file='dictionary_clean.txt', max_distance=2):
        print(f"Loading dictionary: {dictionary_file}...")
        self.dictionary = self.load_dictionary(dictionary_file)
        self.max_distance = max_distance
        print(f"✓ Loaded {len(self.dictionary):,} words")
    
    def load_dictionary(self, filepath):
        """Load dictionary words"""
        words = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = unicodedata.normalize('NFC', line.strip())
                if word:
                    words.add(word)
        return words
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein distance"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        return dp[m][n]
    
    def find_candidates(self, word: str, top_k: int = 5):
        """Find closest words in dictionary"""
        word = unicodedata.normalize('NFC', word)
        
        if word in self.dictionary:
            return [(word, 0)]
        
        candidates = []
        for dict_word in self.dictionary:
            distance = self.edit_distance(word, dict_word)
            if distance <= self.max_distance:
                candidates.append((dict_word, distance))
        
        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]
    
    def correct(self, word: str) -> Optional[str]:
        """Correct a single word"""
        candidates = self.find_candidates(word, top_k=1)
        
        if candidates and candidates[0][1] <= self.max_distance:
            return candidates[0][0]
        
        return None

# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(
    title="Edit-Distance Hindi Spell Checker API",
    description="Pure Levenshtein algorithm with dictionary lookup",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global checker
checker = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SpellCheckRequest(BaseModel):
    text: str

class WordCorrection(BaseModel):
    original: str
    corrected: str
    changed: bool
    edit_distance: Optional[int] = None

class SpellCheckResponse(BaseModel):
    input: str
    corrected: str
    changed: bool
    words: List[WordCorrection]

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global checker
    
    print("=" * 80)
    print("STARTING EDIT-DISTANCE HINDI SPELL CHECKER API")
    print("=" * 80)
    
    checker = LevenshteinSpellChecker(
        dictionary_file='dictionary_clean.txt',
        max_distance=2
    )
    
    print("\n" + "=" * 80)
    print("EDIT-DISTANCE API READY - Running on port 8002")
    print("=" * 80)
    print("\nAlgorithm: Levenshtein (Dynamic Programming)")
    print(f"Dictionary: {len(checker.dictionary):,} words")
    print(f"Max distance: {checker.max_distance}")
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
        "name": "Edit-Distance Hindi Spell Checker API",
        "version": "1.0.0",
        "method": "Levenshtein Algorithm",
        "dictionary_size": f"{len(checker.dictionary):,} words" if checker else "Not loaded",
        "max_distance": checker.max_distance if checker else 2,
        "accuracy": "~55%",
        "port": 8002
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
            corrected = checker.correct(word)
            
            if corrected and corrected != word:
                distance = checker.edit_distance(word, corrected)
                corrected_words.append(corrected)
                word_details.append(WordCorrection(
                    original=word,
                    corrected=corrected,
                    changed=True,
                    edit_distance=distance
                ))
            else:
                corrected_words.append(word)
                word_details.append(WordCorrection(
                    original=word,
                    corrected=word,
                    changed=False,
                    edit_distance=0
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
