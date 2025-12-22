"""
Hybrid Hindi Spell Checker
===========================
Combines Edit-Distance (Levenshtein) with Neural Seq2Seq model
for optimal spelling correction.

Strategy:
1. First: Try edit-distance lookup in dictionary (fast, exact matches)
2. Then: Use neural model for complex corrections (learned patterns)
3. Final: Combine both confidences for best result
"""

import torch
import torch.nn as nn
import unicodedata
from typing import List, Tuple, Optional
import numpy as np


# ============================================================================
# PART 1: EDIT-DISTANCE IMPLEMENTATION
# ============================================================================

class LevenshteinSpellChecker:
    """Traditional edit-distance spell checker"""
    
    def __init__(self, dictionary_file='clean_hindi_words.txt', max_distance=2):
        """
        Args:
            dictionary_file: File with one word per line
            max_distance: Maximum edit distance to consider
        """
        print("Loading dictionary for edit-distance checker...")
        self.dictionary = self.load_dictionary(dictionary_file)
        self.max_distance = max_distance
        print(f"✓ Loaded {len(self.dictionary):,} words")
    
    def load_dictionary(self, filepath):
        """Load dictionary words"""
        words = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = unicodedata.normalize('NFC', line.strip())
                    if word:
                        words.add(word)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Using empty dictionary.")
        return words
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """
        Calculate Levenshtein distance between two words
        
        Returns:
            Integer edit distance (minimum operations needed)
        """
        m, n = len(word1), len(word2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        return dp[m][n]
    
    def find_candidates(self, word: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Find top-k closest words in dictionary
        
        Returns:
            List of (word, distance) tuples sorted by distance
        """
        word = unicodedata.normalize('NFC', word)
        
        # If word is in dictionary, return it
        if word in self.dictionary:
            return [(word, 0)]
        
        # Calculate distances to all dictionary words
        candidates = []
        for dict_word in self.dictionary:
            distance = self.edit_distance(word, dict_word)
            
            # Only consider words within max_distance
            if distance <= self.max_distance:
                candidates.append((dict_word, distance))
        
        # Sort by distance and return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]
    
    def correct(self, word: str) -> Optional[str]:
        """
        Correct a single word using edit-distance
        
        Returns:
            Corrected word or None if no close match found
        """
        candidates = self.find_candidates(word, top_k=1)
        
        if candidates and candidates[0][1] <= self.max_distance:
            return candidates[0][0]
        
        return None


# ============================================================================
# PART 2: NEURAL MODEL (Your Seq2Seq)
# ============================================================================

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


class NeuralSpellChecker:
    """Neural sequence-to-sequence spell checker"""
    
    def __init__(self, model_path='hindi_spelling_model.pt', vocab_path='vocab.txt', device='cpu'):
        """
        Args:
            model_path: Path to trained model
            vocab_path: Path to vocabulary file
            device: 'cpu', 'cuda', or 'mps'
        """
        self.device = torch.device(device)
        
        # Load vocabulary
        print("Loading vocabulary for neural model...")
        self.vocab, self.rev_vocab = self.load_vocab(vocab_path)
        self.PAD_IDX = self.vocab.get("<PAD>", 0)
        self.SOS_IDX = self.vocab.get("<START>", self.vocab.get("<SOS>", 1))
        self.EOS_IDX = self.vocab.get("<END>", self.vocab.get("<EOS>", 2))
        self.UNK_IDX = self.vocab.get("<UNK>", 3)
        print(f"✓ Loaded vocabulary: {len(self.vocab)} characters")
        
        # Initialize model
        print("Loading neural model...")
        attention = Attention(512)
        self.encoder = Encoder(len(self.vocab), 256, 512, self.PAD_IDX).to(self.device)
        self.decoder = Decoder(len(self.vocab), 256, 512, attention, self.PAD_IDX).to(self.device)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Try different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'encoder' in checkpoint and 'decoder' in checkpoint:
                    # Format 1: Separate encoder/decoder
                    self.encoder.load_state_dict(checkpoint['encoder'])
                    self.decoder.load_state_dict(checkpoint['decoder'])
                    print("✓ Loaded encoder/decoder from checkpoint")
                elif 'model_state_dict' in checkpoint:
                    # Format 2: Single model state
                    print("Warning: Checkpoint has model_state_dict - trying to load")
                    # This won't work with separate encoder/decoder
                    raise ValueError("Incompatible checkpoint format")
                else:
                    # Format 3: Direct state dict
                    print("Warning: Unknown checkpoint format")
                    raise ValueError("Incompatible checkpoint format")
            else:
                # Checkpoint is raw state dict - won't work with separate models
                print("Warning: Checkpoint is raw state dict")
                raise ValueError("Incompatible checkpoint format")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("\nThe neural model requires a properly trained checkpoint.")
            print("Please ensure hindi_spelling_model.pt contains encoder/decoder weights.")
            print("\nFalling back to edit-distance only mode...")
            self.encoder = None
            self.decoder = None
        
        self.encoder.eval()
        self.decoder.eval()
        print("✓ Neural model loaded")
    
    def load_vocab(self, vocab_path):
        """Load character vocabulary"""
        vocab = {}
        rev_vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                char = line.strip()
                vocab[char] = idx
                rev_vocab[idx] = char
        return vocab, rev_vocab
    
    def correct(self, word: str, max_len: int = 50) -> str:
        """
        Correct a word using neural model
        
        Returns:
            Corrected word
        """
        # If model didn't load, return original word
        if self.encoder is None or self.decoder is None:
            return word
            
        word = unicodedata.normalize('NFC', word)
        
        with torch.no_grad():
            # Encode word
            src_ids = [self.SOS_IDX] + [self.vocab.get(c, self.UNK_IDX) for c in word] + [self.EOS_IDX]
            src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Encode
            encoder_outputs, hidden, cell = self.encoder(src)
            
            # Decode
            input_token = torch.tensor([self.SOS_IDX], device=self.device)
            decoded_ids = []
            
            for _ in range(max_len):
                output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
                pred = output.argmax(dim=1).item()
                
                if pred == self.EOS_IDX:
                    break
                
                decoded_ids.append(pred)
                input_token = torch.tensor([pred], device=self.device)
        
        # Convert to string
        chars = [
            self.rev_vocab.get(i, "") 
            for i in decoded_ids 
            if i not in (self.PAD_IDX, self.SOS_IDX, self.EOS_IDX)
        ]
        
        return "".join(chars)


# ============================================================================
# PART 3: HYBRID SYSTEM (Combines Both)
# ============================================================================

class HybridSpellChecker:
    """
    Hybrid spell checker combining edit-distance and neural approaches
    
    Strategy:
    1. Check if word exists in dictionary (edit-distance = 0)
    2. Try edit-distance for close matches (distance ≤ 2)
    3. Use neural model for complex corrections
    4. Return best result based on confidence
    """
    
    def __init__(
        self,
        model_path='hindi_spelling_model.pt',
        vocab_path='vocab.txt',
        dictionary_path='clean_hindi_words.txt',
        device='cpu'
    ):
        """Initialize both spell checkers"""
        print("=" * 80)
        print("INITIALIZING HYBRID SPELL CHECKER")
        print("=" * 80)
        
        # Initialize edit-distance checker
        print("\n[1/2] Edit-Distance Component:")
        self.edit_checker = LevenshteinSpellChecker(dictionary_path, max_distance=2)
        
        # Initialize neural checker
        print("\n[2/2] Neural Component:")
        self.neural_checker = NeuralSpellChecker(model_path, vocab_path, device)
        
        print("\n" + "=" * 80)
        print("HYBRID SYSTEM READY")
        print("=" * 80)
    
    def correct(self, word: str, strategy: str = 'hybrid') -> dict:
        """
        Correct a word using specified strategy
        
        Args:
            word: Input word to correct
            strategy: 'edit-distance', 'neural', or 'hybrid'
        
        Returns:
            Dictionary with correction results
        """
        word = unicodedata.normalize('NFC', word)
        
        result = {
            'input': word,
            'edit_distance_correction': None,
            'neural_correction': None,
            'final_correction': word,
            'method_used': 'none',
            'confidence': 0.0
        }
        
        # Strategy 1: Edit-Distance
        if strategy in ['edit-distance', 'hybrid']:
            ed_correction = self.edit_checker.correct(word)
            result['edit_distance_correction'] = ed_correction
        
        # Strategy 2: Neural Model
        if strategy in ['neural', 'hybrid']:
            neural_correction = self.neural_checker.correct(word)
            result['neural_correction'] = neural_correction
        
        # For non-hybrid strategies, return early
        if strategy == 'edit-distance':
            if result['edit_distance_correction'] and result['edit_distance_correction'] != word:
                result['final_correction'] = result['edit_distance_correction']
                result['method_used'] = 'edit-distance'
                result['confidence'] = 0.9
            return result
        
        if strategy == 'neural':
            if result['neural_correction'] and result['neural_correction'] != word:
                result['final_correction'] = result['neural_correction']
                result['method_used'] = 'neural'
                result['confidence'] = 0.7
            return result
        
        # Strategy 3: Hybrid Decision (Smart!)
        if strategy == 'hybrid':
            ed = result['edit_distance_correction']
            neural = result['neural_correction']
            
            # CASE 1: Both agree on a change
            if ed and neural and ed == neural and ed != word:
                result['final_correction'] = ed
                result['method_used'] = 'both-agree'
                result['confidence'] = 0.95
                return result
            
            # CASE 2: Both return the original word (no change needed)
            if (ed == word or ed is None) and (neural == word or neural is None):
                result['final_correction'] = word
                result['method_used'] = 'both-unchanged'
                result['confidence'] = 0.90
                return result
            
            # CASE 3: Edit-distance makes no change, neural suggests change
            if (ed == word or ed is None) and neural and neural != word:
                # Word is in dictionary unchanged, but neural wants to change
                
                # Calculate how much neural changed
                len_diff = len(neural) - len(word)
                
                # Check if neural is making a reasonable change
                if len_diff == 0:
                    # Same length - likely matra substitution (भारतय → भारतीय)
                    # Trust neural
                    result['final_correction'] = neural
                    result['method_used'] = 'neural-override'
                    result['confidence'] = 0.80
                    
                elif len_diff == 1:
                    # Added 1 character - could be valid (अच्छ → अच्छा)
                    # Check if it's adding a matra or vowel
                    added_char = neural[-1] if len(neural) > len(word) else ''
                    matras = 'ािीुूृॄेैोौंःाअआइईउऊऋएऐओऔ'
                    
                    if added_char in matras:
                        # Adding valid Hindi vowel/matra - trust neural
                        result['final_correction'] = neural
                        result['method_used'] = 'neural-override'
                        result['confidence'] = 0.78
                    else:
                        # Adding consonant/other - be cautious
                        result['final_correction'] = word
                        result['method_used'] = 'keep-original'
                        result['confidence'] = 0.75
                        
                elif len_diff > 1:
                    # Adding multiple characters (भारत → भारतत)
                    # Likely wrong - keep original
                    result['final_correction'] = word
                    result['method_used'] = 'keep-original'
                    result['confidence'] = 0.85
                    
                else:  # len_diff < 0
                    # Neural is removing characters
                    # This is often correct (removing extra matras)
                    result['final_correction'] = neural
                    result['method_used'] = 'neural-override'
                    result['confidence'] = 0.77
                    
                return result
            
            # CASE 4: Neural makes no change, edit-distance suggests change
            if (neural == word or neural is None) and ed and ed != word:
                # Neural thinks it's fine, but dictionary has close match
                result['final_correction'] = ed
                result['method_used'] = 'edit-distance-only'
                result['confidence'] = 0.80
                return result
            
            # CASE 5: Both suggest changes but disagree
            if ed and neural and ed != neural and ed != word and neural != word:
                # Calculate edit distances to determine which is closer
                ed_dist = self.edit_checker.edit_distance(word, ed)
                neural_dist = self.edit_checker.edit_distance(word, neural)
                
                if ed_dist < neural_dist:
                    # Edit-distance suggestion is closer to original
                    result['final_correction'] = ed
                    result['method_used'] = 'edit-distance-closer'
                    result['confidence'] = 0.70
                elif neural_dist < ed_dist:
                    # Neural suggestion is closer
                    result['final_correction'] = neural
                    result['method_used'] = 'neural-closer'
                    result['confidence'] = 0.65
                else:
                    # Same distance - PREFER NEURAL (it's trained on correct data)
                    result['final_correction'] = neural
                    result['method_used'] = 'neural-tiebreaker'
                    result['confidence'] = 0.72
                return result
            
            # CASE 6: Only one has a suggestion (shouldn't happen, but handle it)
            if ed and ed != word:
                result['final_correction'] = ed
                result['method_used'] = 'edit-distance-fallback'
                result['confidence'] = 0.75
            elif neural and neural != word:
                result['final_correction'] = neural
                result['method_used'] = 'neural-fallback'
                result['confidence'] = 0.70
        
        return result
    
    def correct_text(self, text: str, strategy: str = 'hybrid') -> str:
        """
        Correct all words in a text
        
        Args:
            text: Input text
            strategy: Correction strategy
        
        Returns:
            Corrected text
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            result = self.correct(word, strategy=strategy)
            corrected_words.append(result['final_correction'])
        
        return " ".join(corrected_words)
    
    def compare_methods(self, word: str) -> dict:
        """
        Compare all three strategies on a word
        
        Returns:
            Dictionary with results from each method
        """
        word = unicodedata.normalize('NFC', word)
        
        comparison = {
            'input': word,
            'edit_distance': self.edit_checker.correct(word),
            'neural': self.neural_checker.correct(word),
            'hybrid': self.correct(word, strategy='hybrid')['final_correction']
        }
        
        return comparison


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize hybrid system
    checker = HybridSpellChecker(
        model_path='hindi_spelling_model.pt',
        vocab_path='vocab.txt',
        dictionary_path='clean_hindi_words.txt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "=" * 80)
    print("TESTING HYBRID SPELL CHECKER")
    print("=" * 80)
    
    # Test cases
    test_words = [
        "भारतय",      # य→ी (neural works well)
        "सरकारि",     # ि→ी (neural works well)
        "भारत",       # correct word (edit-distance finds it)
        "विदयालय",    # missing halant (neural handles)
        "पानि",       # ि→ी (both should work)
    ]
    
    print("\nTest Words:")
    print("-" * 80)
    
    for word in test_words:
        result = checker.correct(word, strategy='hybrid')
        
        print(f"\nInput: {word}")
        print(f"  Edit-Distance: {result['edit_distance_correction']}")
        print(f"  Neural:        {result['neural_correction']}")
        print(f"  Final:         {result['final_correction']}")
        print(f"  Method:        {result['method_used']}")
        print(f"  Confidence:    {result['confidence']:.1%}")
    
    print("\n" + "=" * 80)
    print("Method Comparison:")
    print("=" * 80)
    
    for word in test_words[:3]:
        comparison = checker.compare_methods(word)
        print(f"\n{comparison['input']:15s} → ED: {comparison['edit_distance'] or 'None':15s} | "
              f"Neural: {comparison['neural']:15s} | Hybrid: {comparison['hybrid']:15s}")
    
    print("\n" + "=" * 80)
