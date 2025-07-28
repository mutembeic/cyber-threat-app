# backend/main.py

import json
import os
import re
import string
from pathlib import Path

import nltk
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cyber Threat Detector API",
    version="1.0.0",
    description="An API to classify text as malicious or benign using a trained Keras model."
)

# --- Constants and Paths ---
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent 
APP_DIR = Path(__file__).resolve(strict=True).parent # Path to the 'backend' directory
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "improved_base_model.h5"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
MAX_SEQ_LEN = 250

# --- Pydantic Model for Input Validation ---
class TextInput(BaseModel):
    text: str

# --- Global Variables ---
model = None
tokenizer = None
lemmatizer = None
stop_words = None

# --- Startup Event to Load Resources ---
@app.on_event("startup")
def load_resources():
    """
    Load the trained model, tokenizer, and NLTK data at application startup.
    """
    global model, tokenizer, lemmatizer, stop_words

    # --- NLTK Data Handling ---
    nltk_data_dir = APP_DIR / "nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(str(nltk_data_dir))
    
    print("Downloading NLTK resources to local app directory...")
    nltk.download('wordnet', download_dir=str(nltk_data_dir), quiet=True)
    nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=True)
    nltk.download('punkt', download_dir=str(nltk_data_dir), quiet=True)
    nltk.download('punkt_tab', download_dir=str(nltk_data_dir), quiet=True)

    print("Loading model and tokenizer...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "r") as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise RuntimeError(f"Could not load ML model resources: {e}")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    print("Resources loaded successfully.")

# --- Text Preprocessing Function ---
def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses raw text to match the format used for model training.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Cyber Threat Detector API is running."}

@app.post("/predict/", tags=["Prediction"])
def predict(payload: TextInput):
    """
    Predicts if a given text is malicious or benign.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded. API is not ready.")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        cleaned_text = preprocess_text(payload.text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
        
        prediction_prob = model.predict(padded_sequence)[0][0]
        
        is_malicious = prediction_prob >= 0.5
        label = "Malicious" if is_malicious else "Benign"
        
        return {
            "text": payload.text,
            "prediction": label,
            "is_malicious": bool(is_malicious),
            "confidence": float(prediction_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
