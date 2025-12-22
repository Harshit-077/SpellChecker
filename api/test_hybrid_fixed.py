"""
Test Hybrid System with Split Checkpoint
"""

from hybrid_spell_checker import HybridSpellChecker
import torch

print("=" * 80)
print("TESTING HYBRID SPELL CHECKER (with split checkpoint)")
print("=" * 80)

# Initialize with split checkpoint
checker = HybridSpellChecker(
    model_path='hindi_spelling_model_split.pt',  # ← New split version
    vocab_path='vocab.txt',
    dictionary_path='dictionary_clean.txt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("\n" + "=" * 80)
print("TEST CASES")
print("=" * 80)

test_words = [
    "भारतय",      # य→ी
    "सरकारि",     # ि→ी
    "भारत",       # correct
    "विदयालय",    # missing halant
    "पानि",       # ि→ी
]

for word in test_words:
    result = checker.correct(word, strategy='hybrid')
    
    print(f"\nInput: {word}")
    print(f"  Edit-Distance: {result['edit_distance_correction']}")
    print(f"  Neural:        {result['neural_correction']}")
    print(f"  Final:         {result['final_correction']}")
    print(f"  Method:        {result['method_used']}")
    print(f"  Confidence:    {result['confidence']:.1%}")

print("\n" + "=" * 80)
