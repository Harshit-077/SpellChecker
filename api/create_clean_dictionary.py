"""
Create Clean Dictionary from Training Data
===========================================
Extracts only correct spellings from hindi_pairs.csv
"""

import pandas as pd
from collections import Counter

print("=" * 80)
print("CREATING CLEAN DICTIONARY FROM TRAINING DATA")
print("=" * 80)

# Load training data
print("\nLoading hindi_pairs.csv...")
df = pd.read_csv('../data/hindi_pairs.csv')
print(f"✓ Loaded {len(df):,} training pairs")

# Extract all clean (correct) words
print("\nExtracting clean words...")
clean_words = df['clean'].dropna().unique()
print(f"✓ Found {len(clean_words):,} unique correct words")

# Count frequencies (optional - to keep only common ones)
word_counts = Counter(df['clean'].dropna())

# Sort by frequency (most common first)
sorted_words = sorted(clean_words, key=lambda w: word_counts[w], reverse=True)

# Save dictionary
output_file = 'dictionary_clean.txt'
print(f"\nSaving to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for word in sorted_words:
        f.write(word + '\n')

print(f"✓ Saved {len(sorted_words):,} words")

# Statistics
print("\n" + "=" * 80)
print("DICTIONARY STATISTICS")
print("=" * 80)

lengths = [len(w) for w in sorted_words]
print(f"\nTotal words: {len(sorted_words):,}")
print(f"Average length: {sum(lengths)/len(lengths):.1f} characters")
print(f"Shortest: {min(lengths)} chars")
print(f"Longest: {max(lengths)} chars")

print(f"\nMost common words (top 20):")
for i, (word, count) in enumerate(word_counts.most_common(20), 1):
    print(f"  {i:2d}. {word:15s} (appears {count:,} times)")

print("\n" + "=" * 80)
print("DICTIONARY READY!")
print("=" * 80)
print(f"\nNext step: Use dictionary_clean.txt in hybrid system")
print(f"  dictionary_path='dictionary_clean.txt'")
print("=" * 80)
