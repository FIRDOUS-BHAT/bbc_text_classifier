# train.py
import json
import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# -- Load dataset (try a couple of names)
dataset = None
for name in ("bbc", "bbc_news"):
    try:
        dataset = load_dataset(name, split="train")
        print("Loaded dataset:", name)
        break
    except Exception as e:
        dataset = None

if dataset is None:
    raise RuntimeError(
        "Could not load BBC dataset. Provide CSV or correct HF dataset id.")

df = pd.DataFrame(dataset)

# Infer column names
text_col = 'text' if 'text' in df.columns else [
    c for c in df.columns if 'text' in c.lower()][0]
label_col = 'label' if 'label' in df.columns else [
    c for c in df.columns if c.lower() in ('category', 'label', 'class')][0]

X = df[text_col].astype(str).values
y = df[label_col].values

# Map labels to strings if they're ints
labels = sorted(list(set(y)))
label_to_name = {int(i): str(lbl) for i, lbl in enumerate(
    labels)}  # not used if y already strings
# But we will store label names in order used by the model
unique_labels = sorted(list(set(y)))
label_list = [str(l) for l in unique_labels]
with open(os.path.join(OUT_DIR, "labels.json"), "w") as fh:
    json.dump(label_list, fh)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipeline
tfidf = TfidfVectorizer(stop_words='english', max_df=0.9,
                        min_df=2, ngram_range=(1, 2), max_features=20000)
# Choose model: logistic or linear svc
model_choice = "logistic"  # change to "svc" to use LinearSVC
if model_choice == "logistic":
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
else:
    clf = LinearSVC(max_iter=2000)

pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", clf)
])

print("Training pipeline ...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", acc)
print("Weighted F1:", f1)
print("Classification report:\n", classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(pipeline.named_steps['tfidf'],
            os.path.join(OUT_DIR, "vectorizer.pkl"))
joblib.dump(pipeline.named_steps['clf'], os.path.join(OUT_DIR, "model.pkl"))

# Save full pipeline (optional)
joblib.dump(pipeline, os.path.join(OUT_DIR, "pipeline_full.pkl"))
print("Saved artifacts to", OUT_DIR)
