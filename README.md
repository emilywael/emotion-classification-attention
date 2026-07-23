# 🎭 Emotion Classification with Attention

A multi-class emotion classifier that detects **6 core emotions** — Joy, Sadness, Anger, Fear, Surprise, and Disgust — from free text, comparing classical sequence models against a fine-tuned transformer.

**Live demo:** *(add your Gradio/HF Spaces link here after deploying)*

---

## 🎯 Project Overview

| | |
|---|---|
| **Dataset** | [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Google, 58k Reddit comments) |
| **Task** | 6-class single-label emotion classification |
| **Embeddings** | Pre-trained GloVe (100d) |
| **Models compared** | LSTM · GRU · BiLSTM + Attention · Fine-tuned DistilBERT |
| **Interpretability** | Attention-weight heatmaps over input tokens |
| **Deployment** | Interactive Gradio app |

## 🧠 Architecture Summary

```
Raw GoEmotions (27 labels)
        │
        ▼
Label mapping → 6 target emotions
        │
        ▼
Cleaning + Tokenization + Padding
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
GloVe Embedding                DistilBERT Tokenizer
   │                                  │
   ├─ LSTM                            ▼
   ├─ GRU                     Fine-tuned DistilBERT
   └─ BiLSTM + Attention              │
        │                             │
        └──────────┬──────────────────┘
                    ▼
     Accuracy · Macro F1 · Confusion Matrix · Attention Heatmaps
                    │
                    ▼
              Gradio Deployment
```

## 📊 Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| 🥇 DistilBERT (fine-tuned) | **82.0%** | **71.1%** |
| 🥈 BiLSTM + Attention | 65.9% | 58.9% |
| 🥉 LSTM | 64.3% | 56.5% |
| GRU | 50.6% | 35.5% |

The fine-tuned transformer outperforms all custom sequence models by a wide margin (~12 points of macro F1 over the next best), which is expected given DistilBERT's large-scale pre-training. Among the custom architectures, BiLSTM + Attention edges out plain LSTM, and both clearly beat GRU on this dataset/training setup.

Class weighting (`sklearn.compute_class_weight`, capped at 5.0) was used during training to counter GoEmotions' strong imbalance (joy alone makes up 56.6% of the mapped training set, vs. under 2% for fear/disgust).

*(Full numbers regenerated as `model_comparison_results.csv` when you run the notebook.)*

## 📁 Repo Structure

```
emotion-classification-attention/
├── Emotion_Classification_Notebook.ipynb   # full pipeline: data → training → evaluation → attention viz
├── app.py                                  # Gradio deployment app
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Train (Google Colab recommended — GPU)
Open `Emotion_Classification_Notebook.ipynb` in Colab, set runtime to GPU, and run all cells top to bottom. This will:
- Download & preprocess GoEmotions
- Download GloVe 100d embeddings
- Train LSTM, GRU, and BiLSTM+Attention
- Fine-tune DistilBERT via the HuggingFace `Trainer` API
- Evaluate all 4 models and plot attention heatmaps
- Save all artifacts needed for deployment

### 2. Deploy locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## 🏷️ Label Mapping

GoEmotions' 27 fine-grained emotions are grouped into 6 coarse classes (see notebook Section 3 for the exact mapping and rationale). `neutral` and highly ambiguous labels are dropped so every example maps to exactly one target class.

## 🔍 Interpretability

The BiLSTM+Attention model exposes per-token attention weights, visualized as heatmaps to show which words most influenced each prediction (notebook Section 8).

**Example:**
> `"I'm really scared about what might happen next"` → **fear** (99.1% confidence)
> Attention concentrates almost entirely on the word *"scared"*.

> `"This is amazing news, I'm so excited and grateful!"` → **joy** (98.4% confidence)
> Attention spreads across *"amazing"*, *"excited"*, and *"grateful"*.

## 📦 Tech Stack

`TensorFlow/Keras` · `PyTorch` · `HuggingFace Transformers & Datasets` · `scikit-learn` · `Gradio` · `GloVe`

---

*Built as part of an NLP/deep learning coursework project.*
