import streamlit as st
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Load Model + Scaler
# -------------------------
clf = joblib.load("fft_model.pkl")
scaler = joblib.load("fft_scaler.pkl")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# -------------------------
# Helper: Get FFT embedding
# -------------------------
def get_fft_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    signal = np.mean(embeddings, axis=1)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)

    top_n = 50
    if len(magnitude) >= top_n:
        return magnitude[:top_n]
    else:
        return np.pad(magnitude, (0, top_n - len(magnitude)), mode='constant')

# -------------------------
# Prediction function
# -------------------------
def predict_custom(text):
    vec = get_fft_embedding(text)
    vec_scaled = scaler.transform([vec])
    prob = clf.predict_proba(vec_scaled)[0][1]
    return f" AI Probability: {prob * 100:.2f}% (0 = Human, 100 = AI)"

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI vs Human Text Classifier")
st.write("Paste any paragraph to check if it's AI generated or human written. note: The text sample must contain more than 30 words atleast for better prediction.")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_custom(user_input)
        st.success(result)
    else:
        st.warning("Please enter some text to analyze.")
