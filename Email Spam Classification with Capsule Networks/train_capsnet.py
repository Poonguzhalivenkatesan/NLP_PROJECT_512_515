import pandas as pd
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Capsule Network Implementation
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, dim_capsule, input_dim, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.weight = nn.Parameter(torch.randn(num_capsules, input_dim, dim_capsule))
        
    def squash(self, s, dim=-1):
        norm = torch.norm(s, p=2, dim=dim, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        u_hat = torch.matmul(x, self.weight)  # [batch_size, num_caps, dim_caps]
        
        b = torch.zeros(x.size(0), self.num_capsules, 1, device=x.device)
        for i in range(self.routings):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2)
            v = self.squash(s)
            if i < self.routings - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1, keepdim=True)
        return v

class CapsNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CapsNet, self).__init__()
        self.capsule = CapsuleLayer(num_capsules=num_classes, dim_capsule=16, input_dim=input_dim)
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.capsule(x)
        x = self.fc(x)
        return x

def train():
    # Load and preprocess data
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['text'] = df['text'].apply(preprocess_text)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['text']).toarray()
    y = LabelEncoder().fit_transform(df['label'])
    
    # Save tokenizer
    os.makedirs('model', exist_ok=True)
    with open('model/tokenizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    # Initialize model
    model = CapsNet(input_dim=X_train.shape[1], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_acc = (torch.argmax(val_outputs, 1) == y_test).float().mean()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'num_classes': 2
    }, 'model/capsnet.pth')

if __name__ == '__main__':
    train()