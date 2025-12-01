# load clean sentences from your corpus
sentences = []
with open("data/all_hindi_clean.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # ignore empty lines
            sentences.append(line)
print(f"Loaded {len(sentences)} sentences")

import random
import regex as re

def create_typos(sentence, typo_prob=0.1):
    """
    Randomly introduce typos in a sentence.
    typo_prob: probability of changing a word
    """
    words = sentence.split()
    new_words = []
    for w in words:
        if random.random() < typo_prob:
            # Choose typo type: deletion, substitution, transpose
            typo_type = random.choice(["delete", "replace", "transpose"])
            if typo_type == "delete" and len(w) > 1:
                i = random.randint(0, len(w)-1)
                w = w[:i] + w[i+1:]
            elif typo_type == "replace" and len(w) > 0:
                i = random.randint(0, len(w)-1)
                w = w[:i] + random.choice(list(w)) + w[i+1:]  # replace with random char in word
            elif typo_type == "transpose" and len(w) > 1:
                i = random.randint(0, len(w)-2)
                w = w[:i] + w[i+1] + w[i] + w[i+2:]
        new_words.append(w)
    return " ".join(new_words)

# Example
for _ in range(5):
    sent = random.choice(sentences)
    print("Original:", sent)
    print("With typos:", create_typos(sent))
    print()

# Create dataset
data_pairs = []
for sent in sentences:
    incorrect_sent = create_typos(sent, typo_prob=0.2)
    data_pairs.append((incorrect_sent, sent))  # (input, target)

print("Sample pair:", data_pairs[0])


from collections import Counter

# Count words in target sentences
word_counter = Counter()
for _, target in data_pairs:
    word_counter.update(target.split())

# Special tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# Build vocab (top 50k words)
top_k = 100000
vocab = {PAD:0, SOS:1, EOS:2, UNK:3}
for word, _ in word_counter.most_common(top_k):
    vocab[word] = len(vocab)

# Reverse vocab
rev_vocab = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def sentence_to_ids(sentence, vocab):
    return [vocab.get(word, vocab[UNK]) for word in sentence.split()]

def ids_to_sentence(ids, rev_vocab):
    words = []
    for idx in ids:
        if idx == vocab[EOS]:
            break
        words.append(rev_vocab.get(idx, UNK))
    return " ".join(words)

# Example
inp_ids = sentence_to_ids(data_pairs[0][0], vocab)
tgt_ids = sentence_to_ids(data_pairs[0][1], vocab)
print("Input IDs :", inp_ids)
print("Target IDs:", tgt_ids)

import pickle

# After creating data_pairs
with open("data/data_pairs.pkl", "wb") as f:
    pickle.dump(data_pairs, f)
print("data_pairs saved!")


# Save vocab to file
with open("vocab/hindi_vocab_100k.tsv", "w", encoding="utf-8") as f:
    for word, idx in vocab.items():
        f.write(f"{word}\t{idx}\n")
print("Vocabulary saved!")


