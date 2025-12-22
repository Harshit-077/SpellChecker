"""
Realistic Hindi Noise Generator
Creates training data that matches actual spelling mistakes
"""

import random
import unicodedata
import pandas as pd

# Define error types based on real Hindi mistakes
MATRAS = ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', '्']

PHONETIC_CONFUSIONS = {
    'ब': ['व'], 'व': ['ब'],
    'श': ['स', 'ष'], 'ष': ['स', 'श'], 'स': ['श', 'ष'],
    'न': ['ण'], 'ण': ['न'],
    'त': ['थ'], 'थ': ['त'],
    'द': ['ध'], 'ध': ['द'],
    'प': ['फ'], 'फ': ['प'],
    'क': ['ख'], 'ख': ['क'],
    'ज': ['झ'], 'झ': ['ज'],
    'ि': ['ी'], 'ी': ['ि'],
    'ु': ['ू'], 'ू': ['ु'],
    'े': ['ै'], 'ै': ['े'],
    'ो': ['ौ'], 'ौ': ['ो'],
}


def remove_matra(word):
    """
    ERROR TYPE 1: Remove a matra (40% of errors)
    This is the MOST COMMON real-world error!
    
    Example: भारत → भरत (removed आ)
    """
    chars = list(word)
    matra_positions = [i for i, c in enumerate(chars) if c in MATRAS]
    
    if not matra_positions:
        return word
    
    # Remove a random matra
    idx = random.choice(matra_positions)
    chars.pop(idx)
    return ''.join(chars)


def wrong_matra(word):
    """
    ERROR TYPE 2: Replace matra with wrong one (20% of errors)
    
    Example: भारतय → भारतीय (य→ी)
    """
    chars = list(word)
    matra_positions = [i for i, c in enumerate(chars) if c in MATRAS and c in PHONETIC_CONFUSIONS]
    
    if not matra_positions:
        return word
    
    idx = random.choice(matra_positions)
    old_matra = chars[idx]
    chars[idx] = random.choice(PHONETIC_CONFUSIONS[old_matra])
    return ''.join(chars)


def remove_halant(word):
    """
    ERROR TYPE 3: Remove halant (15% of errors)
    
    Example: विद्यालय → विदयालय (removed ्)
    """
    chars = list(word)
    halant_positions = [i for i, c in enumerate(chars) if c == '्']
    
    if not halant_positions:
        return word
    
    idx = random.choice(halant_positions)
    chars.pop(idx)
    return ''.join(chars)


def substitute_consonant(word):
    """
    ERROR TYPE 4: Phonetic confusion (15% of errors)
    
    Example: शिक्षा → सिक्षा (श→स)
    """
    chars = list(word)
    consonant_positions = [
        i for i, c in enumerate(chars) 
        if c in PHONETIC_CONFUSIONS and c not in MATRAS
    ]
    
    if not consonant_positions:
        return word
    
    idx = random.choice(consonant_positions)
    old_char = chars[idx]
    chars[idx] = random.choice(PHONETIC_CONFUSIONS[old_char])
    return ''.join(chars)


def add_extra_matra(word):
    """
    ERROR TYPE 5: Add duplicate matra (10% of errors)
    
    Example: भारत → भाारत (duplicated ा)
    """
    chars = list(word)
    
    # Find consonants (safe positions to add matras after)
    consonant_positions = [
        i for i, c in enumerate(chars)
        if '\u0915' <= c <= '\u0939'  # Consonant range
    ]
    
    if not consonant_positions:
        return word
    
    idx = random.choice(consonant_positions)
    matra = random.choice(['ा', 'ि', 'ी', 'ु', 'ू'])
    chars.insert(idx + 1, matra)
    return ''.join(chars)


def generate_realistic_noise(word):
    """
    Apply ONE realistic error to a word
    
    Distribution based on actual spelling mistakes:
    - 40% Missing matra
    - 20% Wrong matra
    - 15% Missing halant
    - 15% Phonetic confusion
    - 10% Extra matra
    """
    if len(word) < 2:
        return word
    
    word = unicodedata.normalize('NFC', word)
    
    # Choose error type based on realistic distribution
    rand = random.random()
    
    operations = [
        (0.40, remove_matra),      # 40%
        (0.60, wrong_matra),        # 20%
        (0.75, remove_halant),      # 15%
        (0.90, substitute_consonant), # 15%
        (1.00, add_extra_matra),    # 10%
    ]
    
    for threshold, operation in operations:
        if rand < threshold:
            noisy = operation(word)
            if noisy != word:
                return noisy
            break
    
    # If no change, return original
    return word


def generate_comprehensive_dataset(
    clean_words_file='clean_hindi_words.txt',
    output_file='hindi_realistic_errors.csv',
    num_samples=300000
):
    """
    Generate realistic training data
    """
    print("=" * 80)
    print("GENERATING REALISTIC HINDI SPELLING ERRORS")
    print("=" * 80)
    
    # Load clean words
    print(f"\nLoading {clean_words_file}...")
    with open(clean_words_file, 'r', encoding='utf-8') as f:
        clean_words = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(clean_words):,} clean words")
    
    # Generate noisy pairs
    print(f"\nGenerating {num_samples:,} realistic error pairs...")
    
    noisy_list = []
    clean_list = []
    
    stats = {
        'removed_matra': 0,
        'wrong_matra': 0,
        'removed_halant': 0,
        'substituted': 0,
        'added_matra': 0,
        'unchanged': 0
    }
    
    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{num_samples:,}...")
        
        clean = random.choice(clean_words)
        noisy = generate_realistic_noise(clean)
        
        if noisy == clean:
            stats['unchanged'] += 1
        else:
            # Categorize error type (rough estimate)
            if len(noisy) < len(clean):
                if '्' in clean and '्' not in noisy:
                    stats['removed_halant'] += 1
                else:
                    stats['removed_matra'] += 1
            elif len(noisy) > len(clean):
                stats['added_matra'] += 1
            else:
                # Could be substitution
                stats['wrong_matra'] += 1
            
            noisy_list.append(noisy)
            clean_list.append(clean)
    
    print(f"\n✓ Generated {len(noisy_list):,} valid pairs")
    
    # Statistics
    print("\n" + "=" * 80)
    print("ERROR TYPE DISTRIBUTION")
    print("=" * 80)
    total = sum(stats.values())
    
    for error_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {error_type:20s}: {count:,} ({pct:.1f}%)")
    
    # Show samples
    print("\n" + "=" * 80)
    print("SAMPLE ERRORS (first 20)")
    print("=" * 80)
    print("\nNoisy (with error) → Clean (correct)")
    print("-" * 80)
    
    for i in range(min(20, len(noisy_list))):
        noisy = noisy_list[i]
        clean = clean_list[i]
        print(f"{noisy:25s} → {clean:25s}")
    
    # Verify quality
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    matras_in_clean = sum(1 for w in clean_list if any(c in MATRAS for c in w))
    matras_in_noisy = sum(1 for w in noisy_list if any(c in MATRAS for c in w))
    
    print(f"\nClean words with matras: {matras_in_clean:,} ({matras_in_clean/len(clean_list)*100:.1f}%)")
    print(f"Noisy words with matras: {matras_in_noisy:,} ({matras_in_noisy/len(noisy_list)*100:.1f}%)")
    print(f"\nDifference: {matras_in_clean - matras_in_noisy:,} matras removed (GOOD!)")
    
    # Save
    df = pd.DataFrame({
        'noisy': noisy_list,
        'clean': clean_list
    })
    
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ Saved {len(df):,} pairs")
    
    print("\n" + "=" * 80)
    print("DATASET READY FOR TRAINING!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. mv {output_file} hindi_pairs.csv")
    print(f"  2. python create_vocab.py")
    print(f"  3. python hindi_spelling_corrector_improved.py")
    print("=" * 80)


if __name__ == "__main__":
    generate_comprehensive_dataset(
        clean_words_file='clean_hindi_words.txt',
        output_file='hindi_realistic_errors.csv',
        num_samples=300000  # 300k samples for better coverage
    )
