## Finalized App Instructions

1. Install: `poetry install` or `pip install -r requirements.txt`.
2. EDA: `jupyter notebook notebooks/eda_bbc.ipynb` (run all cells).
3. Train: `python train.py` (outputs artifacts).
4. Run App: `uvicorn api.app:app --reload` → Open http://localhost:8000.
5. Test: Paste text in UI → Get prediction + prob bars.
6. API: POST /predict with form 'text=...' → JSON response.

Notes:

- Handles 'label' → 'category' auto.
- For full BBC: Convert raw folder to CSV in EDA notebook.
- Model: Logistic Regression + TF-IDF (switch to LinearSVC if needed).

Troubleshoot: Ensure paths match (e.g., data in notebooks/data/).
