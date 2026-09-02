import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

df = pd.read_csv("data/master_clauses.csv")

contract_texts = []
txt_folder = "data/full_contract_txt"
for filename in df['Filename']:
    base = filename.replace('.pdf', '.txt')
    full_path = os.path.join(txt_folder, base)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_texts.append(f.read()[:3000])
    else:
        contract_texts.append("")

df['contract_text'] = contract_texts
df_valid = df[df['contract_text'].str.len() > 100].copy()

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df_valid['contract_text'])

target_clauses = [
    'Non-Compete', 'Exclusivity', 'Termination For Convenience',
    'Change Of Control', 'Anti-Assignment', 'Ip Ownership Assignment',
    'License Grant', 'Cap On Liability', 'Uncapped Liability',
    'Audit Rights', 'Insurance', 'Liquidated Damages',
    'Warranty Duration', 'Non-Disparagement', 'No-Solicit Of Employees'
]

os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/legal_tfidf.pkl")
print("Saved TF-IDF")

saved = 0
for clause in target_clauses:
    col = f"{clause}-Answer"
    if col not in df_valid.columns:
        continue
    y = (df_valid[col].astype(str).str.lower() == 'yes').astype(int)
    if y.sum() < 5:
        continue
    model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))
    model.fit(X, y)
    filename = clause.lower().replace(' ', '_').replace('/', '_') + "_model.pkl"
    joblib.dump(model, f"models/{filename}")
    print(f"Saved: {filename}")
    saved += 1

print(f"\nTotal models saved: {saved}")