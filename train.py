import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Paths (relative to root)
DATA_PATH = 'notebooks/data/bbc_sample.csv'
MODEL_DIR = 'notebooks/model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} articles from {DATA_PATH}.")

# Auto-rename
text_col = None
cat_col = None
for col in df.columns:
    if any(word in col.lower() for word in ['text', 'content', 'article', 'body']):
        text_col = col
    if any(word in col.lower() for word in ['category', 'label', 'topic', 'class']):
        cat_col = col

if text_col and text_col != 'text':
    df = df.rename(columns={text_col: 'text'})
if cat_col and cat_col != 'category':
    df = df.rename(columns={cat_col: 'category'})

df = df.dropna(subset=['text', 'category'])
print("Final columns:", df.columns.tolist())
print("Categories:", df['category'].unique())

# Prepare
X = df['text']
y = df['category']

# Safe split: Check balance for stratified
class_counts = y.value_counts()
min_count = class_counts.min()
print(f"Class counts: {class_counts.to_dict()}")
print(f"Min samples per class: {min_count}")

if min_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print("Used stratified split.")
else:
    print("Min samples too low for stratify—using random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# Train & Evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"Model saved to {MODEL_PATH}")

# Update/create labels
categories = sorted(y.unique().tolist())
labels_info = {'categories': categories}
with open(LABELS_PATH, 'w') as f:
    json.dump(labels_info, f, indent=2)
print(f"Labels saved to {LABELS_PATH}: {categories}")
