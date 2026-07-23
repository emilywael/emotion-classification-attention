"""
Emotion Classification — Gradio Deployment App
================================================
Loads the fine-tuned DistilBERT model (default) and serves an interactive demo:
user enters free text -> app returns the dominant emotion + confidence bar chart
across all 6 classes (joy, sadness, anger, fear, surprise, disgust).

Run:
    python app.py

Requires the artifacts produced by the training notebook to sit next to this file:
    ./distilbert_emotion_final/   (HF model + tokenizer, from Section 9/11 of the notebook)
    label_map.json                (from Section 11)

To deploy one of the custom Keras models (LSTM / GRU / BiLSTM+Attention) instead,
set MODEL_BACKEND = "keras" below and make sure lstm_model.h5 / tokenizer.pickle exist.
"""

import json
import pickle

import gradio as gr
import numpy as np

MODEL_BACKEND = "distilbert"  # "distilbert" or "keras"
MAX_LEN = 40

with open("label_map.json") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😮", "disgust": "🤢",
}

# ------------------------------------------------------------------
# Load model (DistilBERT branch)
# ------------------------------------------------------------------
if MODEL_BACKEND == "distilbert":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_PATH = "./distilbert_emotion_final"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    def predict_probs(text: str) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        return probs

# ------------------------------------------------------------------
# Load model (Keras branch — LSTM / GRU / BiLSTM+Attention)
# ------------------------------------------------------------------
else:
    import re
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    KERAS_MODEL_PATH = "bilstm_attention_model.h5"  # swap for lstm_model.h5 / gru_model.h5
    keras_model = load_model(KERAS_MODEL_PATH, compile=False)

    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict_probs(text: str) -> np.ndarray:
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        preds = keras_model.predict(padded, verbose=0)
        return preds[0]


# ------------------------------------------------------------------
# Inference function used by the Gradio UI
# ------------------------------------------------------------------
def classify_emotion(text: str):
    if not text or not text.strip():
        return "—", {}

    probs = predict_probs(text)
    top_idx = int(np.argmax(probs))
    top_label = idx2label[top_idx]
    top_emoji = EMOTION_EMOJI.get(top_label, "")

    label_display = f"{top_emoji}  {top_label.capitalize()}  ({probs[top_idx]:.1%} confidence)"
    confidence_dict = {idx2label[i]: float(probs[i]) for i in range(len(probs))}

    return label_display, confidence_dict


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
demo = gr.Interface(
    fn=classify_emotion,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Type a sentence, e.g. 'I can't believe you did that, I'm furious!'",
        label="Your text",
    ),
    outputs=[
        gr.Label(label="Dominant Emotion"),
        gr.Label(label="Confidence across all 6 classes", num_top_classes=6),
    ],
    title="🎭 Emotion Classifier — Attention-based NLP",
    description=(
        "Classifies free text into one of 6 core emotions: "
        "Joy, Sadness, Anger, Fear, Surprise, Disgust. "
        "Powered by a fine-tuned DistilBERT model trained on GoEmotions."
    ),
    examples=[
        "I just got accepted into my dream university!",
        "I lost my phone and I don't know what to do.",
        "How dare you talk to me like that.",
        "There's a spider the size of my hand on the wall.",
        "Wait, what? I didn't expect that at all.",
        "This smells absolutely disgusting.",
    ],
    theme=gr.themes.Soft(primary_hue="purple"),
)

if __name__ == "__main__":
    demo.launch()
