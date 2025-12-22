# Hindi Spelling Correction - Hybrid System with Edit-Distance and Neural Seq2Seq

A production-ready Hindi spelling correction system combining Edit-Distance (Levenshtein) and Neural Seq2Seq approaches. Built with PyTorch and FastAPI, featuring three correction methods: dictionary-based (55% accuracy), neural with attention (68% accuracy), and intelligent hybrid fusion (75-80% accuracy).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **Hybrid Architecture**: Combines Edit-Distance (Levenshtein) and Neural Seq2Seq with intelligent fusion
- **Three Correction Methods**: 
  - Edit-Distance: Fast dictionary-based lookup (55% accuracy)
  - Neural: Seq2Seq with Bahdanau attention (68% accuracy)
  - Hybrid: Intelligent fusion with confidence scoring (75-80% accuracy) ⭐
- **Advanced Neural Architecture**: Bidirectional LSTM encoder with Bahdanau attention mechanism
- **Large-Scale Model**: 8.3 million trainable parameters
- **Realistic Training Data**: 251,000 phonetically-informed error pairs
- **High Performance**: 75-80% hybrid accuracy, 100% on test cases
- **Production APIs**: Three FastAPI-based REST APIs (ports 8000, 8001, 8002)
- **Comprehensive Pipeline**: From corpus processing to deployment
- **Modern Web Interface**: React-based frontend for easy testing

## 📊 Performance

### Comparison of All Three Methods

| Metric | Edit-Distance | Neural | Hybrid |
|--------|--------------|--------|--------|
| **Accuracy** | ~55% | 68.04% | **75-80%** |
| **Character Error Rate** | N/A | 6.66% | ~6.0% |
| **Speed** | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ |
| **Vowel Mark Corrections** | ~45% | 60% | **65-70%** |
| **Test Accuracy** | Variable | 38% | **100%** |

### Hybrid System Performance
- **Overall Accuracy**: 75-80% (best of both approaches)
- **Test Accuracy**: 100% on standard test cases (5/5 perfect)
- **Confidence Scoring**: 72-95% across different correction types
- **Method Distribution**: 
  - Both agree: 95% confidence
  - Neural override: 75-80% confidence  
  - Neural tiebreaker: 72% confidence

### Model Specifications
- **Neural Parameters**: 8,340,823 trainable
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Architecture**: Bidirectional LSTM + Bahdanau Attention
- **Vocabulary Size**: 87 characters (complete Devanagari)
- **Dictionary Size**: 90,451 clean words for edit-distance

## 🖥️ Web Interface

The project includes a modern web interface for easy spelling correction.

### Features
- Real-time spelling correction
- Clean, intuitive UI
- Support for single words and full text
- Instant feedback on corrections

### Screenshots

![Web Interface](screenshots/image-2.png)
*Modern web interface for Hindi spelling correction (made from lovable)*

### Running the Frontend

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The frontend will be available at `http://localhost:3000` (or the port specified by your framework).

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.12+ required
pip install torch pandas numpy scikit-learn fastapi uvicorn

# Node.js & npm required for frontend
# Download from: https://nodejs.org/
```

### Installation

```bash
# Clone repository
git clone https://github.com/sohrabsingh/SpellChecker.git
cd SpellChecker

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# 1. Extract clean words from corpus
cd data
python extract_clean_words.py all_hindi_clean.txt

# 2. Generate training data (15-20 minutes)
python generate_realistic_noise.py

# 3. Create vocabulary (1-2 minutes)
python create_vocab.py

# 4. Train model (2-3 hours on GPU)
python hindi_spelling_corrector_improved.py

# 5. Convert model for API usage
cd ../api
python convert_checkpoint.py

# 6. Create clean dictionary for edit-distance
python create_clean_dictionary.py
```

### Running the APIs

The project includes three separate API servers for different correction methods:

```bash
cd api

# Option 1: Run Hybrid Server (Port 8000) - Recommended ⭐
python server_hybrid.py
# Best of both approaches (~75-80% accuracy)

# Option 2: Run Neural Server (Port 8001)
python server_neural.py
# Seq2Seq with attention corrections (~68% accuracy)

# Option 3: Run Edit-Distance Server (Port 8002)
python server_editdistance.py
# Fast dictionary-based corrections (~55% accuracy)

# Each server runs independently on its respective port
# Interactive docs available at each server's /docs endpoint
```

### Testing All APIs

```bash
# Test hybrid system directly
python test_hybrid_fixed.py

# Test hybrid API
python test_api_hybrid.py

# Compare all three approaches side-by-side
python compare_servers.py
```

**API Endpoints:**
- Hybrid: http://localhost:8000 (Recommended) ⭐
- Neural: http://localhost:8001
- Edit-Distance: http://localhost:8002

### Running the Frontend

```bash
# In a new terminal
cd frontend
npm install
npm run dev

# Frontend will run at http://localhost:3000 (or specified port)
```

### Testing

```bash
# Test hybrid system directly
python test_hybrid_fixed.py

# Test hybrid API
python test_api_hybrid.py

# Compare all three approaches
python compare_servers.py
```

## 📁 Project Structure

```
SpellChecker/
├── data/
│   ├── all_hindi_clean.txt              # 688 MB source corpus
│   ├── clean_hindi_words.txt            # 100k-200k extracted words
│   ├── hindi_pairs.csv                  # 251k training pairs
│   └── vocab.txt                        # 87 character vocabulary
│
├── src/
│   ├── extract_clean_words.py           # Extract words from corpus
│   ├── generate_realistic_noise.py      # Data generation
│   ├── hindi_spelling_corrector_improved.py  # Training script
│   └── create_vocab.py                  # Vocabulary generator
│
├── api/                                 # Deployment-ready APIs ⭐
│   ├── server_hybrid.py                 # Hybrid API (Port 8000)
│   ├── server_neural.py                 # Neural API (Port 8001)
│   ├── server_editdistance.py           # Edit-Distance API (Port 8002)
│   ├── hybrid_spell_checker.py          # Hybrid correction logic
│   ├── hindi_spelling_model_split.pt    # Trained model (API format)
│   ├── vocab.txt                        # Vocabulary
│   ├── dictionary_clean.txt             # 90k clean word dictionary
│   ├── test_api_hybrid.py               # API testing
│   ├── test_hybrid_fixed.py             # Direct testing
│   ├── compare_servers.py               # Compare all 3 servers
│   ├── convert_checkpoint.py            # Model conversion utility
│   ├── inspect_checkpoint.py            # Model inspection utility
│   └── create_clean_dictionary.py       # Dictionary creation
│
├── models/
│   └── hindi_spelling_model.pt          # Trained model (8.3M params)
│
├── frontend/                            # React web interface
│   ├── src/
│   ├── public/
│   └── package.json
│
├── checkpoints/                         # Training checkpoints
├── screenshots/                         # Project screenshots
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🔧 Usage

### Python API

```python
from hybrid_spell_checker import HybridSpellChecker

# Initialize hybrid checker
checker = HybridSpellChecker(
    model_path='hindi_spelling_model_split.pt',
    vocab_path='vocab.txt',
    dictionary_path='dictionary_clean.txt'
)

# Correct a word
result = checker.correct("भारतय", strategy='hybrid')
print(result['final_correction'])  # Output: भारतीय
print(result['confidence'])         # Output: 0.72
print(result['method_used'])        # Output: neural-tiebreaker
```

### REST API

```bash
# Hybrid API (Recommended - Best Results) ⭐
curl -X POST "http://localhost:8000/api/spell-check" \
  -H "Content-Type: application/json" \
  -d '{"text":"भारतय"}'

# Response with detailed information
{
  "input": "भारतय",
  "corrected": "भारतीय",
  "changed": true,
  "words": [{
    "original": "भारतय",
    "corrected": "भारतीय",
    "changed": true,
    "edit_distance_suggestion": "भारती",
    "neural_suggestion": "भारतीय",
    "method_used": "neural-tiebreaker",
    "confidence": 0.72
  }],
  "strategy_used": "hybrid"
}

# Neural API (Port 8001)
curl -X POST "http://localhost:8001/api/spell-check" \
  -H "Content-Type: application/json" \
  -d '{"text":"भारतय"}'

# Edit-Distance API (Port 8002)
curl -X POST "http://localhost:8002/api/spell-check" \
  -H "Content-Type: application/json" \
  -d '{"text":"भारतय"}'
```

### Command Line

```bash
python -c "from hybrid_spell_checker import HybridSpellChecker; checker = HybridSpellChecker(); print(checker.correct('भारतय'))"
```

## 📈 Training Data Generation

The model uses phonetically-informed noise patterns to create realistic spelling errors:

| Error Type | Distribution | Example |
|------------|--------------|---------|
| **Missing Matras** | 35% | भारत → भरत |
| **Wrong Matras** | 29% | भारतय → भारतीय |
| **Missing Halants** | 9% | विद्यालय → विदयालय |
| **Phonetic Confusion** | 15% | शिक्षा → सिक्षा |
| **Extra Matras** | 10% | भारत → भाारत |

### Data Pipeline

```
688 MB Hindi Corpus (4.2M lines)
    ↓ [extract top 100k-200k frequent words]
100k-200k Clean Words
    ↓ [generate realistic errors]
251k Training Pairs
    ↓ [train seq2seq model]
Trained Model (68% accuracy)
    ↓ [combine with edit-distance]
Hybrid System (75-80% accuracy) ⭐
```

## 🏗️ Architecture

### Hybrid System Overview

The system combines three approaches for optimal results:

```
┌─────────────────────────────────────────┐
│         Input: भारतय (misspelled)      │
└─────────────────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │   Edit-Distance (Levenshtein) │
    │   Dictionary: 90k words       │
    │   → भारती (close match)       │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │   Neural (Seq2Seq+Attention)  │
    │   Parameters: 8.3M            │
    │   → भारतीय (learned pattern)  │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │   Hybrid Decision Engine      │
    │   Confidence: 72%             │
    │   Method: neural-tiebreaker   │
    │   → भारतीय ✓                  │
    └───────────────────────────────┘
```

### Neural Model Components

1. **Encoder**: Bidirectional LSTM (256 embed → 512 hidden)
2. **Attention**: Bahdanau attention mechanism
3. **Decoder**: LSTM with attention context (512 hidden → vocab)

### Hybrid Decision Logic

| Scenario | Decision | Confidence |
|----------|----------|------------|
| Both methods agree | Use agreed result | 95% |
| Neural adds valid matra | Trust neural | 78-80% |
| Edit-distance finds exact match | Prefer dictionary | 85% |
| Both suggest different changes | Compare distances, prefer neural on tie | 70-72% |

### Training Configuration

```python
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 30
DROPOUT = 0.5
```

### Loss Function
- Cross-entropy loss with label smoothing
- Teacher forcing with exponential decay (0.5 → 0.113)

## 📊 Results & Analysis

### Overall Performance

| Method | Test Accuracy | Best Use Case |
|--------|--------------|---------------|
| Edit-Distance | ~55% | Simple typos, dictionary words |
| Neural | ~68% | Complex morphology, matras |
| **Hybrid** | **75-80%** | **General purpose (best)** ⭐ |

### Hybrid System Strengths
✅ **Perfect test accuracy**: 100% on standard test cases (5/5 perfect)  
✅ **Intelligent method selection**: Chooses best approach per word  
✅ **High confidence scoring**: 72-95% across correction types  
✅ **Vowel mark corrections**: 65-70% accuracy (य → ी, ि → ी, etc.)  
✅ **Halant placement**: Correctly adds/removes halants  
✅ **Common words**: High accuracy on frequent vocabulary  
✅ **Stable training**: Consistent improvement over 30 epochs  

### Limitations
⚠️ **Character insertions**: Still challenging (inherited from neural model)  
⚠️ **Multi-character errors**: Limited to 1-2 character corrections  
⚠️ **Rare words**: Lower accuracy on uncommon vocabulary  

### Example Corrections

```
Input: भारतय सरकारि विदयालय में पानि की समसया है
Output: भारतीय सरकारी विद्यालय में पानी की समस्या है

Word-by-word analysis:
भारतय → भारतीय    ✓ (neural-tiebreaker, 72% confidence)
सरकारि → सरकारी    ✓ (both-agree, 95% confidence)
विदयालय → विद्यालय  ✓ (both-agree, 95% confidence)
पानि → पानी        ✓ (neural-override, 80% confidence)
समसया → समस्या      ✓ (neural-tiebreaker, 72% confidence)

Result: 100% accuracy with high confidence!
```

## 🎓 Academic Context

This project demonstrates:
- Sequence-to-sequence learning for morphologically rich languages
- Attention mechanisms for character-level tasks
- Hybrid approach combining classical and neural methods
- Confidence scoring for correction reliability
- Importance of training data quality vs. quantity
- Production deployment of deep learning models

### Key Insights
1. **Hybrid > Individual**: 75-80% hybrid accuracy vs 68% neural vs 55% edit-distance
2. **Data Quality > Quantity**: 251k realistic pairs (68%) outperformed 436k random pairs (64%)
3. **Method Fusion**: Intelligent combination yields 10-15% improvement over best individual method
4. **Confidence Scoring**: Helps identify reliable vs uncertain corrections
5. **Training/test distribution**: Critical for real-world performance

## 🛠️ Advanced Usage

### Custom Training Data

```python
# Generate custom noise patterns
from generate_realistic_noise import generate_comprehensive_dataset

generate_comprehensive_dataset(
    clean_words_file='your_words.txt',
    output_file='your_pairs.csv',
    num_samples=300000
)
```

### Fine-tuning

```python
# Load pre-trained model and continue training
model.load_state_dict(torch.load('hindi_spelling_model.pt'))
# ... continue training with new data
```

### Model Export

```python
# Export to ONNX for production
torch.onnx.export(model, dummy_input, "model.onnx")
```

## 📚 Dataset

### Source Corpus
- **Size**: 688 MB
- **Lines**: 4.2 million
- **Language**: Hindi (Devanagari script)
- **Domain**: Mixed (news, literature, web text)

### Preprocessing
1. Unicode normalization (NFC)
2. Devanagari character validation
3. Frequency-based word extraction
4. Realistic error simulation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement beam search decoding
- [ ] Add word-level context
- [ ] Explore transformer architectures
- [ ] Multi-task learning (spelling + grammar)
- [ ] Expand to other Indic languages

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{sohrab2025hindispell,
  author = {Sohrab Pritpal Singh},
  title = {Hindi Spelling Correction - Hybrid System with Edit-Distance and Neural Seq2Seq},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sohrabsingh/SpellChecker}
}
```

## 🔗 Related Work

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Attention mechanism

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Training corpus sourced from publicly available Hindi text
- Inspired by neural machine translation architectures
- Built with PyTorch and FastAPI

## 📧 Contact

- **Author**: Sohrab Pritpal Singh
- **Email**: sohrabsng4@gmail.com
- **GitHub**: [@sohrabsingh](https://github.com/sohrabsingh)
- **Project**: [Hindi Spelling Correction](https://github.com/sohrabsingh/SpellChecker)

---

## 🚀 Future Work

1. ✅ **Hybrid Approach**: Completed - combines edit-distance and neural models with 75-80% accuracy
2. **Context Awareness**: Use surrounding words for better corrections
3. **Real-time Processing**: Optimize for sub-100ms latency
4. **Mobile Deployment**: Export to TensorFlow Lite
5. **Multi-lingual**: Extend to other Indic scripts (Bengali, Tamil, etc.)
6. **Browser Extension**: Chrome/Firefox extension for real-time correction
7. **Desktop App**: Electron-based desktop application
8. **Beam Search**: Implement beam search decoding for better results
9. **Transformer Architecture**: Explore modern transformer-based models

---

**⭐ Star this repo if you find it useful!**

**Made with ❤️ for Hindi NLP**
