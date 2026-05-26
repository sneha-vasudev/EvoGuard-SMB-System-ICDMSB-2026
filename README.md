# EvoGuard-SMB: Autonomous Neuroevolutionary Email Intelligence Agent

**Author:** Sneha Vasudev  
**Institution:** VIT Bhopal University  
**Contact:** cvsneha.vasudev@gmail.com

## Overview

EvoGuard-SMB is an **autonomous neuroevolutionary multi-task email intelligence agent** designed specifically for small and medium businesses (SMBs) in India. It simultaneously addresses email security threats and customer intelligence by:

- 🛡️ **Threat Detection**: Classifying emails as safe, spam, or phishing
- 💬 **Sentiment Analysis**: Understanding customer emotions (negative, neutral, positive)
- 📧 **Intent Classification**: Identifying communication intent (informational, complaint, purchase)
- 🚨 **Urgency Routing**: Flagging high-priority messages for immediate attention

### Key Innovation: NEAT-based Architecture

Unlike traditional static ML models, EvoGuard-SMB uses **NeuroEvolution of Augmenting Topologies (NEAT)** to:
- **Evolve neural network architectures dynamically** tailored to each business's unique email patterns
- **Balance accuracy, efficiency, and diversity** with a novel 3-component fitness function
- **Operate as a fully autonomous agent** that quarantines threats, routes priorities, and logs CRM updates
- **Deploy on resource-constrained infrastructure** with zero IT expertise required

## Results

| Metric | Performance |
|--------|-------------|
| Threat Detection F1 | **89.6%** |
| Sentiment Analysis F1 | **90.5%** |
| Intent Classification F1 | **90.6%** |
| Urgency Detection F1 | **99.99%** |
| Inference Latency | **0.40 ms** |
| Architecture Search Time | **7.0 minutes** |

Competitive with deep learning baselines while maintaining:
- 📉 Low computational overhead
- ⚡ Fast inference (0.40 ms per email)
- 🔄 Adaptive to evolving email patterns
- 💾 Compact model size (~5KB)

## Architecture

```
┌─────────────────────┐
│   Incoming Emails   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│  Data Ingestion & Cleaning  │
└──────────┬──────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Feature Extraction              │
│  - TF-IDF (5000 features)        │
│  - Behavioral Features (7)       │
│  Total: 5007-dimensional vectors │
└──────────┬───────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Multi-Task NEAT Classifier │
│  - Threat Level (3 outputs) │
│  - Sentiment (3 outputs)    │
│  - Intent (3 outputs)       │
│  - Urgency (1 output)       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────┐
│  Autonomous Agent       │
│  - Threat Quarantine    │
│  - Priority Routing     │
│  - CRM Integration      │
│  - Summary Reports      │
└─────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- ~500MB disk space for datasets

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/evoguard-smb.git
cd evoguard-smb
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Place your email datasets in the `database/` folder:
```
database/
├── Enron.csv              # Real business emails
├── TREC-07.csv            # Spam dataset
├── Nazario_5.csv          # Phishing dataset
├── complaints_processed.csv  # Customer feedback
├── Bitext_Sample_Customer_Service_Training.csv
└── Email Analysis Dataset.csv
```

**CSV Format:**
```csv
text,is_spam,is_phishing,sentiment,y_intent,urgent
```

### 2. Run Training Pipeline

```bash
# Full pipeline with NEAT training
python main.py --train-neat --generations 50

# Only baseline models (faster)
python main.py --no-neat

# Custom generations
python main.py --generations 100
```

### 3. Results & Outputs

Training generates:
```
outputs/
├── models/
│   └── neat_model.pkl         # Trained NEAT model
├── quarantine/                # Quarantined emails
├── priority/                  # High-priority emails
├── crm/                       # CRM logs
├── summary_report_*.txt       # Weekly intelligence reports
└── logs/                      # Training logs
```

## Detailed Usage

### Using Individual Modules

#### Data Loading
```python
from data_loader import load_and_prepare_data

# Load all datasets automatically
df, label_mappings = load_and_prepare_data()
print(f"Loaded {len(df)} emails")
```

#### Feature Extraction
```python
from feature_extraction import FeatureEngineer

engineer = FeatureEngineer()
X_train_features, _ = engineer.fit_transform(df_train)
X_test_features, _ = engineer.transform(df_test)

print(f"Features: {X_train_features.shape}")  # (n_samples, 5007)
```

#### Baseline Models
```python
from baselines import BaselineModels

baselines = BaselineModels()
results = baselines.train_all(
    X_train, y_train,
    X_test, y_test,
    models=['logistic_regression', 'random_forest', 'mlp', 'xgboost']
)

baselines.print_summary()
```

#### NEAT Training
```python
from neat_classifier import NEATClassifier
import numpy as np

# Create classifier
neat = NEATClassifier(
    n_inputs=5007,  # Feature dimension
    task_sizes=[3, 3, 3, 1]  # [threat, sentiment, intent, urgency]
)

# Convert to dense (required by NEAT)
X_train_dense = X_train.toarray().astype(np.float32)
X_test_dense = X_test.toarray().astype(np.float32)

# Train
results = neat.train(X_train_dense, Y_train, generations=50)

# Predict
predictions, probabilities = neat.predict(X_test_dense)
```

#### Autonomous Agent
```python
from agent import EmailAgent

agent = EmailAgent()

# Process single email
result = agent.process_email(
    email_id="email_001",
    email_text="Hello, I have an urgent issue...",
    predictions={'threat': 0, 'sentiment': 0, 'intent': 1, 'urgency': 1},
    probabilities={...}
)

# Generate intelligence report
report_path = agent.export_summary_to_file()
```

## Model Performance

### Comparison with Baselines

| Model | Threat F1 | Training Time | Memory |
|-------|-----------|--------------|--------|
| Logistic Regression | 77.91% | 34s | Low |
| Random Forest | 89.02% | 75s | High |
| MLP (Deep) | 93.43% | 841s | High |
| **EvoGuard-SMB (NEAT)** | **89.60%** | 420s | **Low** |
| XGBoost | 91.2% | 180s | High |

### Strengths
✅ **Competitive accuracy** with deep learning models  
✅ **Faster inference** than most baselines (0.40ms)  
✅ **Lower memory footprint** suitable for shared hosting  
✅ **Multi-task learning** captures task interdependencies  
✅ **Adaptive architecture** evolves for specific business patterns  

### Limitations
⚠️ Longer training time for initial evolution  
⚠️ Requires parameter tuning for optimal performance  
⚠️ Needs representative training data for good generalization  

## Configuration

Edit `config.py` to customize:

```python
# Feature extraction
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Multi-task weights
TASK_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # threat, sentiment, intent, urgency

# NEAT evolution
NEAT_CONFIG = {
    'pop_size': 100,           # Population size
    'generations': 50,         # Evolution generations
    'fitness_threshold': 0.95  # Stopping criterion
}

# Agent settings
AUTO_QUARANTINE_ENABLED = True
AUTO_PRIORITY_ROUTING_ENABLED = True
CRM_LOGGING_ENABLED = True
```

## Project Structure

```
evoguard-smb/
├── config.py                   # Configuration settings
├── data_loader.py             # Data loading & preprocessing
├── feature_extraction.py       # TF-IDF & behavioral features
├── baselines.py               # Baseline models (LR, RF, MLP, XGBoost, SVM)
├── neat_classifier.py         # NEAT-based classifier
├── agent.py                   # Autonomous agent logic
├── main.py                    # Main training pipeline
├── utils.py                   # Utility functions
├── requirements.txt           # Dependencies
├── database/                  # Place datasets here
├── outputs/                   # Training results (auto-created)
│   ├── models/
│   ├── quarantine/
│   ├── priority/
│   ├── crm/
│   └── logs/
└── README.md                  # This file
```

## Datasets

### Recommended Datasets

1. **Enron Email Corpus** (enterprise emails)
   - 400k+ real business emails
   - Download: https://www.cs.cmu.edu/~enron/

2. **TREC-2007 Spam Dataset** (spam examples)
   - 75k labeled spam emails
   - Download: https://plg.uwaterloo.ca/gvcormac/trecspam/

3. **Nazario Phishing Dataset** (phishing examples)
   - 10k+ phishing emails
   - Download: https://www.unispambox.com/

4. **Customer Service Datasets**
   - Sentiment140 (Twitter): https://www.kaggle.com/datasets/kazanova/sentiment140
   - GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions

## Fitness Function

EvoGuard-SMB's NEAT evolution uses a novel **3-component fitness function**:

```
Fitness = 0.5 × F1_multi + 0.2 × Novelty - 0.3 × Latency_penalty
```

- **F1_multi** (0.5): Weighted multi-task F1 score (ensures accuracy)
- **Novelty** (0.2): Genetic diversity reward (prevents premature convergence)
- **Latency_penalty** (-0.3): Inference speed penalty (maintains efficiency)

This balances:
- **Accuracy**: High F1 scores across all tasks
- **Diversity**: Population maintains architectural variety
- **Efficiency**: Models remain fast enough for real-time inference

## Feature Engineering

### TF-IDF Features (5000)
- Unigrams and bigrams
- Sublinear term frequency scaling
- Minimum document frequency: 2
- Maximum document frequency: 95%

### Behavioral Features (7)
1. **message_length**: Number of words
2. **num_links**: Count of URLs
3. **has_html**: Presence of HTML tags
4. **has_urgency**: Urgent keywords detection
5. **free_domain**: Free email domain (phishing indicator)
6. **has_promo**: Promotional content
7. **has_cta**: Call-to-action phrases

**Total: 5007-dimensional feature vectors**

## Deployment

### On Shared Hosting

```bash
# Minimal dependencies installation
pip install -r requirements-minimal.txt

# Run inference server
python inference_server.py --port 8000
```

### Docker Deployment

```bash
docker build -t evoguard-smb .
docker run -p 8000:8000 evoguard-smb
```

### API Usage

```python
import requests

# Send email for classification
response = requests.post('http://localhost:8000/classify', json={
    'email_id': 'email_001',
    'text': 'Check out our amazing offer!'
})

print(response.json())
# {
#   'threat': 'spam',
#   'sentiment': 'positive',
#   'intent': 'purchase',
#   'urgency': 'normal',
#   'action': 'quarantine'
# }
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit Pull Request

## Citation

If you use EvoGuard-SMB in your research, please cite:

```bibtex
@article{vasudev2026evoguard,
  title={EvoGuard-SMB: An Autonomous Neuroevolutionary Multi-Task Email Intelligence Agent},
  author={Vasudev, Sneha},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details

## References

### Key Papers

1. Stanley, K. O., & Miikkulainen, R. (2002). "Evolving neural networks through augmenting topologies." *Evolutionary computation*, 10(2), 99-127.

2. Lehman, J., & Stanley, K. O. (2011). "Abandoning objectives: Evolution through the search for novelty alone." *Evolutionary computation*, 19(2), 189-223.

3. Caruana, R. (1997). "Multitask learning." *Machine learning*, 28(1), 41-75.

4. Ramos, J. (2003). "Using TF-IDF to determine word relevance in document queries." *Proceedings of the first instructional conference on machine learning*.

5. Fette, I., Sadeh, N., & Tomasic, A. (2007). "Learning to detect phishing emails." *Proceedings of the 16th international conference on World Wide Web* (pp. 649-656).

## FAQ

**Q: How much training data do I need?**  
A: Minimum 10,000 emails (5,000 per class). Ideally 50,000+ for robust evolution.

**Q: Can I use this for real-time email filtering?**  
A: Yes! Inference latency is 0.40ms, suitable for real-time systems.

**Q: Do I need GPU?**  
A: No. CPU-optimized. Training on modern CPUs takes ~7 minutes.

**Q: How often should I retrain?**  
A: Monthly recommended. NEAT can retrain in minutes on new data.

**Q: Is it production-ready?**  
A: Yes, but requires integration with your email system (SMTP, IMAP, etc.).

## Troubleshooting

### Issue: NEAT convergence too slow
**Solution:** Reduce population size in `config.py` or increase `fitness_threshold`

### Issue: Out of memory during NEAT training
**Solution:** Reduce batch size or convert features to dense format gradually

### Issue: Baseline models fail on large datasets
**Solution:** Use `BASELINE_SAMPLE_SIZE` stratified sampling (already implemented)

### Issue: Low threat detection accuracy
**Solution:** Ensure balanced dataset. Add more phishing/spam examples.

## Roadmap

- [ ] Transformer-based embeddings (BERT) for richer semantic understanding
- [ ] Online learning capability for continuous adaptation
- [ ] Integration with Gmail/Office365 APIs
- [ ] Web dashboard for monitoring and analytics
- [ ] Explainability module (LIME/SHAP integration)
- [ ] Multi-language support

## Support

For issues, questions, or suggestions:
- 📧 Email: cvsneha.vasudev@gmail.com
- 🐛 Bug Reports: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

**Made with ❤️ for India's 63 million SMBs**

*EvoGuard-SMB: Bringing enterprise-grade email intelligence to businesses of all sizes.*
