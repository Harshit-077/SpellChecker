# Hindi Spell Checker (Seq2Seq) 🚀

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)
![License](https://img.shields.io/badge/license-MIT-green)

A PyTorch-based sequence-to-sequence model for **Hindi spell correction**. This project generates typos, builds a vocabulary, trains a Seq2Seq model with scheduled teacher forcing, and evaluates predictions. Optimized for Windows and low-VRAM GPUs (e.g., NVIDIA 3050 4GB).

---

## Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Setup](#setup)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Checkpoint Loading](#optional-checkpoint-loading)  
- [Example Output](#example-output)  
- [License](#license)  
- [Author](#author)  

---

## Features

- Synthetic typo generation for Hindi text.
- Character-level Seq2Seq model (LSTM encoder-decoder).  
- Scheduled teacher forcing during training.  
- Supports GPU acceleration via CUDA.  
- Checkpointing & model saving.  
- Evaluation with token-level accuracy and sample predictions.  
- Windows & low-VRAM friendly.

---

## Project Structure

```

.
├── data/                  # Raw and processed data
│   └── all_hindi_clean.txt
├── vocab/                 # Vocabulary files
├── checkpoints/           # Model checkpoints
├── InitiallyOkay.ipynb    # Source code
├── encoder_state_dict.h5
├── decoder_state_dict.h5
├── README.md
└── requirements.txt

````

---

## Setup

### 1. Clone the repo
```bash
git clone <repo_url>
cd <repo_name>
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure CUDA support (optional)

```python
import torch
torch.cuda.is_available()
```

---

## Usage

### 1. Prepare Data

Place your Hindi corpus in:

```
data/all_hindi_clean.txt
```

The script automatically generates typo-target pairs.

### 2. Train the Model

* Default: 10 epochs
* Batch size: 16 (gradient accumulation simulates 32)
* Scheduled teacher forcing decays from 1.0 → 0.5
* Best checkpoint saved in `checkpoints/seq2seq_best.pt`

### 3. Evaluate & Sample Predictions

* Outputs average loss & token-level accuracy.
* Displays sample input → target → predicted sequences.

---

## Model Details

* **Encoder:** LSTM, embedding + dropout
* **Decoder:** LSTM, embedding + dropout + linear output
* **Loss:** CrossEntropyLoss (ignores `<PAD>`)
* **Optimizer:** Adam
* **Batch size:** 16 (configurable)
* **Gradient accumulation:** 2 steps

---

## Optional Checkpoint Loading

```python
checkpoint = torch.load("checkpoints/seq2seq_epoch3.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
```

---

## Example Output

```
Input     : मने स्कूल जाना हे
Target    : मुझे स्कूल जाना है
Predicted : मुझे स्कूल जाना है
------------------------------------------------------------
Input     : वह बाज़ार मे गया
Target    : वह बाज़ार में गया
Predicted : वह बाज़ार में गया
```

---

## License

MIT License – free for academic and personal use.

---

## Author

**Sohrab Singh**



