import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import joblib
import os
import time

print("=== LegalMind Model Evaluation ===\n")

df = pd.read_csv("data/master_clauses.csv")

# Load contract texts
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

print(f"Dataset: {len(df)} total contracts | {len(df_valid)} with text")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df_valid['contract_text'])

# Focus clause
target_clause = 'Non-Compete-Answer'
y = (df_valid[target_clause].astype(str).str.lower() == 'yes').astype(int)

print(f"\nTarget clause: Non-Compete")
print(f"Positive samples: {y.sum()} | Negative: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare models
models = {
    "LinearSVC": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
print("\nModel Comparison:")
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    start = time.time()
    model.predict(X_test)
    inf_time = (time.time() - start) / X_test.shape[0] * 1000

    results[name] = {
        'accuracy': acc, 'f1': f1,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'train_time': train_time, 'inference_ms': inf_time
    }

    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Train: {train_time:.2f}s | Inference: {inf_time:.3f}ms")

# All clauses evaluation
target_clauses = [
    'Non-Compete-Answer', 'Exclusivity-Answer',
    'Termination For Convenience-Answer', 'Change Of Control-Answer',
    'Anti-Assignment-Answer', 'Ip Ownership Assignment-Answer',
    'Cap On Liability-Answer', 'Uncapped Liability-Answer',
    'Audit Rights-Answer', 'Insurance-Answer',
    'Liquidated Damages-Answer', 'Warranty Duration-Answer',
    'Non-Disparagement-Answer', 'No-Solicit Of Employees-Answer'
]

print("\n\n=== All Clause Evaluation (LinearSVC) ===")
all_accs, all_f1s = [], []

for col in target_clauses:
    y_col = (df_valid[col].astype(str).str.lower() == 'yes').astype(int)
    if y_col.sum() < 5:
        continue
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_col, test_size=0.2, random_state=42, stratify=y_col)
    m = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))
    m.fit(X_tr, y_tr)
    y_p = m.predict(X_te)
    acc = accuracy_score(y_te, y_p)
    f1 = f1_score(y_te, y_p, average='weighted')
    all_accs.append(acc)
    all_f1s.append(f1)
    print(f"{col.replace('-Answer',''):35} Acc: {acc:.3f} | F1: {f1:.3f}")

print(f"\nAverage Accuracy: {np.mean(all_accs):.4f}")
print(f"Average F1 Score: {np.mean(all_f1s):.4f}")

# Plot
os.makedirs("notebooks/plots", exist_ok=True)
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
f1s = [results[n]['f1'] for n in names]
x = np.arange(len(names))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
plt.bar(x + width/2, f1s, width, label='F1 Score', color='orange')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('LegalMind - Model Comparison (Non-Compete Clause)')
plt.xticks(x, names)
plt.ylim(0.5, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/plots/model_comparison.png", dpi=150)
print("\nPlot saved.")

print("\n=== RESUME METRICS ===")
best = max(results, key=lambda x: results[x]['cv_mean'])
print(f"Dataset: 510 real legal contracts (CUAD dataset)")
print(f"Clause types analyzed: {len(all_accs)}")
print(f"Best Model: {best}")
print(f"Average Accuracy across all clauses: {np.mean(all_accs)*100:.1f}%")
print(f"Average F1 Score: {np.mean(all_f1s)*100:.1f}%")
print(f"CV F1 Score: {results[best]['cv_mean']*100:.1f}% (+/- {results[best]['cv_std']*100:.1f}%)")
print(f"Inference Speed: {results[best]['inference_ms']:.3f}ms per contract")
print(f"Models compared: {', '.join(names)}")