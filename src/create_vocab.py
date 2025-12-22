"""
Generate vocabulary from Hindi spelling dataset
"""

import pandas as pd
import unicodedata

def create_vocabulary(csv_file='hindi_pairs.csv', output_file='vocab.txt'):
    """
    Create vocabulary file from Hindi spelling dataset
    """
    print("=" * 80)
    print("GENERATING VOCABULARY")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading {csv_file}...")
    df = pd.read_csv(csv_file).dropna()
    print(f"✓ Loaded {len(df)} pairs")
    
    # Collect all unique characters
    all_chars = set()
    
    for col in range(2):  # Both noisy and clean columns
        for text in df.iloc[:, col].astype(str):
            # Normalize to NFC
            normalized = unicodedata.normalize('NFC', text)
            all_chars.update(normalized)
    
    # Sort characters
    chars_sorted = sorted(all_chars)
    
    print(f"\nFound {len(chars_sorted)} unique characters")
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab = special_tokens + chars_sorted
    
    print(f"Total vocabulary size (with special tokens): {len(vocab)}")
    
    # Show character categories
    print("\nCharacter breakdown:")
    consonants = [c for c in chars_sorted if '\u0915' <= c <= '\u0939']  # क to ह
    vowels = [c for c in chars_sorted if '\u0905' <= c <= '\u0914']  # अ to औ
    matras = [c for c in chars_sorted if '\u093E' <= c <= '\u094F']  # ा to ौ
    other_marks = [c for c in chars_sorted if '\u0901' <= c <= '\u0903']  # ँ ं ः
    virama = [c for c in chars_sorted if c == '\u094D']  # ्
    numbers = [c for c in chars_sorted if '\u0966' <= c <= '\u096F']  # ० to ९
    
    print(f"  Consonants: {len(consonants)}")
    print(f"  Vowels: {len(vowels)}")
    print(f"  Matras (vowel signs): {len(matras)}")
    print(f"  Other marks: {len(other_marks)}")
    print(f"  Virama (्): {len(virama)}")
    print(f"  Numbers: {len(numbers)}")
    print(f"  Other: {len(chars_sorted) - len(consonants) - len(vowels) - len(matras) - len(other_marks) - len(virama) - len(numbers)}")
    
    # Save vocabulary
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char + '\n')
    
    print(f"✓ Vocabulary saved!")
    
    # Show first 30 items
    print("\nFirst 30 vocabulary items:")
    print(vocab[:30])
    
    print("\n" + "=" * 80)
    print("VOCABULARY GENERATION COMPLETE")
    print("=" * 80)
    
    return vocab

if __name__ == "__main__":
    vocab = create_vocabulary()
