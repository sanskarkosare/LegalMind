import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

print("=== LegalMind Clause Classifier Training ===\n")

df = pd.read_csv("data/master_clauses.csv")

# Target clause columns
target_clauses = [
    'Non-Compete-Answer',
    'Exclusivity-Answer',
    'Termination For Convenience-Answer',
    'Change Of Control-Answer',
    'Anti-Assignment-Answer',
    'Ip Ownership Assignment-Answer',
    'License Grant-Answer',
    'Cap On Liability-Answer',
    'Uncapped Liability-Answer',
    'Audit Rights-Answer',
    'Insurance-Answer',
    'Liquidated Damages-Answer',
    'Warranty Duration-Answer',
    'Non-Disparagement-Answer',
    'No-Solicit Of Employees-Answer'
]

# Load contract text
print("Loading contract texts...")
contract_texts = []
txt_folder = "data/full_contract_txt"

for filename in df['Filename']:
    txt_file = os.path.join(txt_folder, filename.replace('.pdf', '.txt').split('_')[0] + '_' + '_'.join(filename.split('_')[1:]).replace('.pdf', '.txt'))
    # Try direct filename match
    base = filename.replace('.pdf', '.txt')
    full_path = os.path.join(txt_folder, base)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_texts.append(f.read()[:3000])
    else:
        contract_texts.append("")

df['contract_text'] = contract_texts
print(f"Loaded {sum(1 for t in contract_texts if t)} contract texts out of {len(df)}")

# Use contracts with text
df_valid = df[df['contract_text'].str.len() > 100].copy()
print(f"Valid contracts with text: {len(df_valid)}")

# TF-IDF features
print("\nExtracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df_valid['contract_text'])
print(f"Feature matrix: {X.shape}")

# Train classifier for each clause
os.makedirs("models", exist_ok=True)
results = {}

print("\nTraining classifiers for each clause type...\n")
for clause in target_clauses:
    y = (df_valid[clause].astype(str).str.lower() == 'yes').astype(int)

    if y.sum() < 5:
        print(f"Skipping {clause} — insufficient positive samples ({y.sum()})")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    results[clause] = {
        'accuracy': acc,
        'f1': f1,
        'cv_f1': cv_scores.mean(),
        'positive_rate': y.mean()
    }

    print(f"{clause.replace('-Answer','')}")
    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | CV F1: {cv_scores.mean():.3f}")

# Save models and tfidf
joblib.dump(tfidf, "models/legal_tfidf.pkl")
print("\nTF-IDF vectorizer saved.")

# Summary
print("\n=== RESULTS SUMMARY ===")
accs = [r['accuracy'] for r in results.values()]
f1s = [r['f1'] for r in results.values()]
print(f"Average Accuracy: {np.mean(accs):.3f}")
print(f"Average F1 Score: {np.mean(f1s):.3f}")
print(f"Best Clause: {max(results, key=lambda x: results[x]['f1'])}")
print(f"Clauses trained: {len(results)}")

# Plot
plt.figure(figsize=(14, 6))
clauses = [c.replace('-Answer','').replace(' ','\\n') for c in results.keys()]
f1_vals = [results[c]['f1'] for c in results.keys()]
plt.bar(clauses, f1_vals, color='steelblue')
plt.axhline(y=np.mean(f1_vals), color='red', linestyle='--', label=f'Mean F1: {np.mean(f1_vals):.3f}')
plt.xlabel('Clause Type')
plt.ylabel('F1 Score')
plt.title('LegalMind - Clause Classification F1 Scores')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.legend()
plt.tight_layout()
os.makedirs("notebooks/plots", exist_ok=True)
plt.savefig("notebooks/plots/clause_f1_scores.png", dpi=150)
print("\nPlot saved to notebooks/plots/clause_f1_scores.png")

# Resume metrics
print("\n=== RESUME METRICS ===")
print(f"Dataset: 510 real legal contracts from CUAD dataset")
print(f"Clause types classified: {len(results)}")
print(f"Average Accuracy: {np.mean(accs)*100:.1f}%")
print(f"Average F1 Score: {np.mean(f1s)*100:.1f}%")
print(f"Features: TF-IDF with {5000} features, bigrams")