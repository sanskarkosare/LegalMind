import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

print("=== LegalMind Dataset Analysis ===\n")

df = pd.read_csv("data/master_clauses.csv")
print(f"Dataset shape: {df.shape}")
print(f"Total contracts: {len(df)}")

# Get Yes/No clause columns
answer_cols = [col for col in df.columns if col.endswith('-Answer') or col.endswith('- Answer')]
print(f"\nTotal clause types: {len(answer_cols)}")
print("\nClause columns:")
for col in answer_cols[:15]:
    yes_count = (df[col].astype(str).str.lower() == 'yes').sum()
    no_count = (df[col].astype(str).str.lower() == 'no').sum()
    print(f"  {col}: Yes={yes_count}, No={no_count}")

# Focus on binary Yes/No clauses for classification
binary_clauses = []
for col in answer_cols:
    vals = df[col].astype(str).str.lower().unique()
    if 'yes' in vals and 'no' in vals:
        binary_clauses.append(col)

print(f"\nBinary clause columns: {len(binary_clauses)}")
print(binary_clauses)

# Save analysis
os.makedirs("notebooks", exist_ok=True)
pd.DataFrame({
    'clause': binary_clauses,
    'yes_count': [(df[c].astype(str).str.lower() == 'yes').sum() for c in binary_clauses],
    'no_count': [(df[c].astype(str).str.lower() == 'no').sum() for c in binary_clauses]
}).to_csv("notebooks/clause_analysis.csv", index=False)

print("\nAnalysis saved to notebooks/clause_analysis.csv")