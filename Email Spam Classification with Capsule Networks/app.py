# app.py
from flask import Flask, render_template, request
import torch
import pickle
import numpy as np
from train_capsnet import preprocess_text, CapsNet  # Now this will work correctly

app = Flask(__name__)

def load_model():
    # Load tokenizer
    with open('model/tokenizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    # Initialize model architecture
    model = CapsNet(input_dim=5000, num_classes=2)  # Adjust dimensions as needed
    
    # Load weights
    model.load_state_dict(torch.load('model/capsnet.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, tfidf

model, tfidf = load_model()

# ... (rest of your Flask routes remain the same)