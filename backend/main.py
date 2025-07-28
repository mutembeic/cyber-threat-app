import pickle
import re
import string
from pathlib import Path

import nltk
import tensorflow as tf
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Initialize the FastAPI app
app = FastAPI(title="Cyber Threat Detector API", version="1.0.0")

# Define constants
MAX_SEQ_LEN = 250
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
MODEL_PATH = BASE_DIR / "model" / "improved_base_model.h5"   
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.json"

#Pydantic Model for Input Validation
class TextInput(BaseModel):
    text: str

#Global Variables for Model and Tokenizer 
# These will be loaded at startup
model = None
tokenizer = None
lemmatizer = None
stop_words = None

#Startup Event to Load Models and NLTK data 
@app.on_event("startup")
def load_resources():
    global model, tokenizer, lemmatizer, stop_words

    print("Loading model and tokenizer...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    print("Loading NLTK resources...")
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    print("Resources loaded successfully.")

#Text Preprocessing Function 
def preprocess_text(text: str) -> str:
    """Replicates the preprocessing from the notebook."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

#API Endpoints
@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Cyber Threat Detector API is running."}

@app.post("/predict/")
def predict(payload: TextInput):
    """
    Predict if a given text is malicious or benign.
    """
    if not payload.text:
        return {"error": "Input text cannot be empty."}

    # Preprocess the input text
    cleaned_text = preprocess_text(payload.text)
    
    # Convert to sequence and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    
    # Make prediction
    prediction_prob = model.predict(padded_sequence)[0][0]
    
    # Determine label
    is_malicious = prediction_prob >= 0.5
    label = "Malicious" if is_malicious else "Benign"
    
    return {
        "text": payload.text,
        "prediction": label,
        "is_malicious": bool(is_malicious),
        "confidence": float(prediction_prob)
    }
