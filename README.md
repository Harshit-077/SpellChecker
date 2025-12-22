# Hindi Spelling Correction using Seq2Seq with Attention

A deep learning-based Hindi spelling correction system built with PyTorch, featuring a sequence-to-sequence architecture with Bahdanau attention mechanism. The model achieves 68% accuracy on realistic spelling errors and is deployed as a production-ready REST API.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **Advanced Architecture**: Bidirectional LSTM encoder with Bahdanau attention mechanism
- **Large-Scale Model**: 8.3 million trainable parameters
- **Realistic Training Data**: 251,000 phonetically-informed error pairs
- **High Performance**: 68% validation accuracy, 60% on vowel mark corrections
- **Production API**: FastAPI-based REST API for easy integration
- **Comprehensive Pipeline**: From corpus processing to deployment

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 68.04% |
| **Character Error Rate (CER)** | 6.66% |
| **Vowel Mark Corrections** | 60% accuracy |
| **Overall Test Accuracy** | 38% on matched distributions |

### Model Specifications
- **Parameters**: 8,340,823 trainable
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Architecture**: Bidirectional LSTM + Attention
- **Vocabulary Size**: 87 characters (complete Devanagari)

## 🖥️ Web Interface

The project includes a modern web interface for easy spelling correction.

### Features
- Real-time spelling correction
- Clean, intuitive UI
- Support for single words and full text
- Instant feedback on corrections

### Screenshots

![web Interface](screenshots/image-2.png)
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
cd src
python extract_clean_words.py all_hindi_clean.txt

# 2. Generate training data (15-20 minutes)
python generate_realistic_noise.py

# 3. Create vocabulary (1-2 minutes)
python create_vocab.py

# 4. Train model (2-3 hours on GPU)
python hindi_spelling_corrector_improved.py
```

### Running the API

```bash
# Start the backend server
cd api
uvicorn server:app --reload --port 8000

# Server will run at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

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
# Test on matched cases (38% accuracy)
python test_cases_that_work.py

# Test API
python test_api.py
```

## 📁 Project Structure

```
hindi-spelling-correction/
├── data/
│   ├── clean_hindi_words.txt            # 100k extracted words
│   ├── hindi_pairs.csv                  # 251k training pairs
│   └── vocab.txt                        # 87 character vocabulary
│
├── src/
│   ├── generate_realistic_noise.py      # Data generation
│   ├── hindi_spelling_corrector_improved.py  # Training script
│   ├── extract_clean_words.py           # Extract clean words from corpus
│   ├── create_vocab.py                  # Vocabulary generator
│   ├── test_cases_that_work.py          # Testing script
│   └── test_api.py                      # API test script
│
├── api/
│   ├── model.py                         # Model architecture
│   └── server.py                        # FastAPI server
│
├── models/
│   └── hindi_spelling_model.pt          # Trained model (8.3M params)
│
├── frontend/                            # Front-end files
├── checkpoints/                         # Training checkpoints (not uploaded due to its size)
├── requirements.txt                     # Python dependencies
├── screenshots/                         # screenshots of the project
└── README.md                            # This file
```

## 🔧 Usage

### Python API

```python
from model import correct_word

# Load model
encoder, decoder = load_model('models/hindi_spelling_model.pt')

# Correct spelling
corrected = correct_word(encoder, decoder, "भारतय", vocab, rev_vocab, device)
print(corrected)  # Output: भारतीय
```

### REST API

```bash
# Correct single word
curl -X POST "http://localhost:8000/api/spell-check" \
  -H "Content-Type: application/json" \
  -d '{"text":"भारतय"}'

# Response
{
  "input": "भारतय",
  "corrected": "भारतीय",
  "changed": true
}
```

### Command Line

```bash
python -c "from model import correct_word; print(correct_word('भारतय'))"
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
    ↓ [extract top 100k frequent words]
100k Clean Words
    ↓ [generate realistic errors]
251k Training Pairs
    ↓ [train seq2seq model]
Trained Model (68% accuracy)
```

## 🏗️ Architecture

### Model Components

1. **Encoder**: Bidirectional LSTM (256 embed → 512 hidden)
2. **Attention**: Bahdanau attention mechanism
3. **Decoder**: LSTM with attention context (512 hidden → vocab)

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

### Strengths
✅ **Vowel mark corrections**: 60% accuracy (य → ी, ि → ी, etc.)  
✅ **Halant placement**: Correctly adds/removes halants  
✅ **Common words**: High accuracy on frequent vocabulary  
✅ **Stable training**: Consistent improvement over 30 epochs  

### Limitations
⚠️ **Character insertions**: Struggles with adding missing vowels  
⚠️ **Multi-character errors**: Limited to 1-2 character corrections  
⚠️ **Rare words**: Lower accuracy on uncommon vocabulary  

### Error Analysis

```
Input: भारतय    → Output: भारतीय    ✓ (vowel mark correction)
Input: सरकारि   → Output: सरकारी    ✓ (vowel mark correction)
Input: विदयालय  → Output: विद्यालय  ✓ (halant correction)
Input: भरत      → Output: भरता      ✗ (needs insertion)
```

## 🎓 Academic Context

This project demonstrates:
- Sequence-to-sequence learning for morphologically rich languages
- Attention mechanisms for character-level tasks
- Importance of training data quality vs. quantity
- Production deployment of deep learning models

### Key Insights
1. **Data Quality > Quantity**: 251k realistic pairs (68%) outperformed 436k random pairs (64%)
2. **Character-level limitations**: Better at substitutions than insertions
3. **Training/test distribution**: Critical for real-world performance

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
@misc{hindi-spelling-correction,
  author = {Sohrab Pritpal Singh},
  title = {Hindi Spelling Correction using Seq2Seq with Attention},
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

1. **Hybrid Approach**: Combine character-level and word-level models
2. **Context Awareness**: Use surrounding words for better corrections
3. **Real-time Processing**: Optimize for sub-100ms latency
4. **Mobile Deployment**: Export to TensorFlow Lite
5. **Multi-lingual**: Extend to other Indic scripts (Bengali, Tamil, etc.)
6. **Browser Extension**: Chrome/Firefox extension for real-time correction
7. **Desktop App**: Electron-based desktop application

---

**⭐ Star this repo if you find it useful!**

**Made with ❤️ for Hindi NLP**
