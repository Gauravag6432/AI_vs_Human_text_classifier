# AI_vs_Human_text_classifier (BERT + FFT + Random Forest)
This project model predicts whether the given text is ai generated or human written. 
note:- it works for large text samples (word count > 30 words at least for better accuracy)

- BERT embeddings for semantic features
- FFT (Fast Fourier Transform) for frequency-domain features
- Random Forest classifier

check out this project deployed at [https://ai-text-classifier.streamlit.app/](https://ai-text-classifier.streamlit.app/)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
