"""
STEP 1: Extract and Validate Clean Hindi Words from Corpus
============================================================
This script reads your 688MB corpus and extracts only valid Hindi words.
"""

import unicodedata
import re
from collections import Counter

def is_valid_hindi_char(char):
    """Check if character is valid Hindi (Devanagari script)"""
    code = ord(char)
    # Devanagari range: U+0900 to U+097F
    return 0x0900 <= code <= 0x097F

def is_valid_hindi_word(word):
    """
    Validate that a word is proper Hindi with correct structure.
    
    Valid Hindi word should:
    - Have at least one consonant or independent vowel
    - Not start with a matra (matras must follow consonants)
    - Have reasonable length (2-50 chars)
    - Only contain Devanagari characters
    """
    if not word or len(word) < 2 or len(word) > 50:
        return False
    
    # Check all characters are Devanagari
    if not all(is_valid_hindi_char(c) for c in word):
        return False
    
    # Normalize to NFC
    word = unicodedata.normalize('NFC', word)
    
    # Matras (vowel signs) - these cannot appear at the start
    matras = 'ािीुूृॄेैोौंः्ॅॉ'
    
    # Word should not start with a matra
    if word[0] in matras:
        return False
    
    # Should have at least one consonant (क-ह) or vowel (अ-औ)
    has_consonant = any('\u0915' <= c <= '\u0939' for c in word)
    has_vowel = any('\u0905' <= c <= '\u0914' for c in word)
    
    if not (has_consonant or has_vowel):
        return False
    
    return True

def extract_clean_words(corpus_file, output_file, min_frequency=2, max_words=100000):
    """
    Extract clean Hindi words from corpus.
    
    Args:
        corpus_file: Path to your 688MB text file
        output_file: Where to save clean words
        min_frequency: Minimum times word must appear (filters typos)
        max_words: Maximum words to extract
    """
    print("=" * 80)
    print("EXTRACTING CLEAN HINDI WORDS FROM CORPUS")
    print("=" * 80)
    
    print(f"\nReading corpus from: {corpus_file}")
    print("This may take a few minutes for a 688MB file...")
    
    word_counts = Counter()
    total_lines = 0
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines, found {len(word_counts):,} unique words...")
                
                # Extract words (split by whitespace and punctuation)
                words = re.findall(r'[\u0900-\u097F]+', line)
                
                for word in words:
                    # Normalize
                    word = unicodedata.normalize('NFC', word.strip())
                    
                    # Validate
                    if is_valid_hindi_word(word):
                        word_counts[word] += 1
                
                total_lines = line_num
    
    except FileNotFoundError:
        print(f"\n✗ Error: File '{corpus_file}' not found!")
        print("\nPlease ensure your corpus file is in the current directory.")
        print("Expected format: Plain text file with Hindi content")
        return None
    
    except Exception as e:
        print(f"\n✗ Error reading file: {e}")
        return None
    
    print(f"\n✓ Processed {total_lines:,} lines")
    print(f"✓ Found {len(word_counts):,} unique valid Hindi words")
    
    # Filter by frequency
    print(f"\nFiltering words that appear at least {min_frequency} times...")
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_frequency}
    
    print(f"✓ {len(filtered_words):,} words meet frequency threshold")
    
    # Get most common words
    most_common = word_counts.most_common(max_words)
    final_words = [word for word, count in most_common]
    
    print(f"\nExtracting top {len(final_words):,} words...")
    
    # Statistics
    lengths = [len(word) for word in final_words]
    avg_length = sum(lengths) / len(lengths)
    
    print(f"\nWord Statistics:")
    print(f"  Total words: {len(final_words):,}")
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Shortest: {min(lengths)} chars")
    print(f"  Longest: {max(lengths)} chars")
    
    # Show samples
    print(f"\nSample words (first 20):")
    for i, word in enumerate(final_words[:20], 1):
        count = word_counts[word]
        print(f"  {i:2d}. {word:20s} (frequency: {count:,})")
    
    # Count matras
    matras = 'ािीुूृॄेैोौंः्'
    total_matras = sum(sum(1 for c in word if c in matras) for word in final_words)
    words_with_matras = sum(1 for word in final_words if any(c in matras for c in word))
    
    print(f"\nMatra Statistics:")
    print(f"  Words with matras: {words_with_matras:,} ({words_with_matras/len(final_words)*100:.1f}%)")
    print(f"  Total matras: {total_matras:,}")
    print(f"  Avg matras per word: {total_matras/len(final_words):.2f}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in final_words:
            f.write(word + '\n')
    
    print(f"✓ Saved {len(final_words):,} clean Hindi words")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\nNext step: Generate noisy versions")
    print(f"  python generate_noisy_data.py")
    print("=" * 80)
    
    return final_words

if __name__ == "__main__":
    import sys
    
    # Check if corpus file is provided
    if len(sys.argv) > 1:
        corpus_file = sys.argv[1]
    else:
        # Default corpus filename (adjust as needed)
        corpus_file = 'hindi_corpus.txt'
        print(f"Using default corpus file: {corpus_file}")
        print(f"To specify a different file: python extract_clean_words.py <your_corpus_file>")
    
    extract_clean_words(
        corpus_file=corpus_file,
        output_file='clean_hindi_words.txt',
        min_frequency=2,      # Word must appear at least 2 times
        max_words=100000      # Extract top 100k words
    )
