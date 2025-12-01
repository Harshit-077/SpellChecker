# Load vocab from file
vocab = {}
with open("vocab/hindi_vocab_100k.tsv", "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split("\t")
        vocab[word] = int(idx)

# Reverse vocab for decoding
rev_vocab = {idx: word for word, idx in vocab.items()}
print("Vocab loaded! Size:", len(vocab))




import torch
from torch.utils.data import Dataset, DataLoader

class HindiSpellDataset(Dataset):
    def __init__(self, pairs, vocab):
        """
        pairs: list of (input_sentence_with_typos, target_correct_sentence)
        vocab: word -> index dictionary
        """
        self.pairs = pairs
        self.vocab = vocab
        self.SOS = vocab["<SOS>"]
        self.EOS = vocab["<EOS>"]
        self.UNK = vocab["<UNK>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # Encode words to IDs
        src_ids = [self.vocab.get(word, self.UNK) for word in src.split()] + [self.EOS]
        tgt_ids = [self.SOS] + [self.vocab.get(word, self.UNK) for word in tgt.split()] + [self.EOS]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)



def collate_fn(batch):
    """
    batch: list of (src_tensor, tgt_tensor)
    returns: padded src, padded tgt, src_lengths, tgt_lengths
    """
    PAD_IDX = vocab["<PAD>"]
    src_batch, tgt_batch = zip(*batch)
    
    # Get max lengths
    src_max_len = max([len(s) for s in src_batch])
    tgt_max_len = max([len(t) for t in tgt_batch])
    
    # Pad sequences
    src_padded = torch.full((len(batch), src_max_len), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), PAD_IDX, dtype=torch.long)
    
    src_lengths = []
    tgt_lengths = []
    
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t
        src_lengths.append(len(s))
        tgt_lengths.append(len(t))
    
    return src_padded, tgt_padded, torch.tensor(src_lengths), torch.tensor(tgt_lengths)

import pickle

with open("data/data_pairs.pkl", "rb") as f:
    data_pairs = pickle.load(f)
print(f"Loaded {len(data_pairs)} sentence pairs")


# Create dataset
dataset = HindiSpellDataset(data_pairs, vocab)

# DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Check one batch
for src, tgt, src_len, tgt_len in dataloader:
    print("SRC shape:", src.shape)
    print("TGT shape:", tgt.shape)
    print("SRC lengths:", src_len[:5])
    print("TGT lengths:", tgt_len[:5])
    break
