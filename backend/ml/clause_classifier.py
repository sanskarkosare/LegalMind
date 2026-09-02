import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

TFIDF_PATH = "models/legal_tfidf.pkl"

RISK_CLAUSES = {
    'Non-Compete': 'HIGH',
    'Uncapped Liability': 'HIGH',
    'Liquidated Damages': 'HIGH',
    'Anti-Assignment': 'MEDIUM',
    'Change Of Control': 'MEDIUM',
    'Ip Ownership Assignment': 'MEDIUM',
    'Cap On Liability': 'LOW',
    'Exclusivity': 'MEDIUM',
    'Termination For Convenience': 'MEDIUM',
    'License Grant': 'LOW',
    'Insurance': 'LOW',
    'Audit Rights': 'LOW',
    'Warranty Duration': 'LOW',
    'Non-Disparagement': 'MEDIUM',
    'No-Solicit Of Employees': 'MEDIUM'
}

CLAUSE_DESCRIPTIONS = {
    'Non-Compete': 'Restricts parties from competing with each other after contract ends.',
    'Uncapped Liability': 'No limit on damages — extremely risky clause.',
    'Liquidated Damages': 'Fixed penalty amount for breach of contract.',
    'Anti-Assignment': 'Prevents transferring contract rights to third parties.',
    'Change Of Control': 'Triggers rights/obligations if company ownership changes.',
    'Ip Ownership Assignment': 'Assigns intellectual property ownership to another party.',
    'Cap On Liability': 'Limits maximum damages — protective clause.',
    'Exclusivity': 'Restricts working with competitors during contract period.',
    'Termination For Convenience': 'Allows termination without cause.',
    'License Grant': 'Grants permission to use IP or technology.',
    'Insurance': 'Requires maintaining insurance coverage.',
    'Audit Rights': 'Allows auditing of financial records.',
    'Warranty Duration': 'Defines how long warranties are valid.',
    'Non-Disparagement': 'Prevents negative statements about the other party.',
    'No-Solicit Of Employees': 'Prevents hiring employees from the other party.'
}

def load_tfidf():
    if os.path.exists(TFIDF_PATH):
        return joblib.load(TFIDF_PATH)
    return None

def calculate_risk_score(detected_clauses: list) -> dict:
    score = 0
    for clause in detected_clauses:
        risk = RISK_CLAUSES.get(clause, 'LOW')
        if risk == 'HIGH':
            score += 3
        elif risk == 'MEDIUM':
            score += 2
        else:
            score += 1

    max_possible = len(detected_clauses) * 3 if detected_clauses else 1
    normalized = (score / max_possible) * 100 if max_possible > 0 else 0

    if normalized >= 60:
        level = "HIGH RISK"
        color = "red"
    elif normalized >= 35:
        level = "MEDIUM RISK"
        color = "orange"
    else:
        level = "LOW RISK"
        color = "green"

    return {
        "score": round(normalized, 1),
        "level": level,
        "color": color,
        "raw_score": score
    }

def analyze_contract(text: str) -> dict:
    tfidf = load_tfidf()
    if tfidf is None:
        return {"error": "Model not loaded. Run training script first."}

    # Load all clause models and predict
    df = pd.read_csv("data/master_clauses.csv")

    target_clauses = list(RISK_CLAUSES.keys())
    answer_cols = [f"{c}-Answer" for c in target_clauses]

    # Load contract texts for training
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

    X_all = tfidf.transform(df_valid['contract_text'])
    X_new = tfidf.transform([text[:3000]])

    detected_clauses = []
    clause_details = []

    for clause in target_clauses:
        col = f"{clause}-Answer"
        if col not in df_valid.columns:
            continue

        y = (df_valid[col].astype(str).str.lower() == 'yes').astype(int)
        if y.sum() < 5:
            continue

        model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))
        model.fit(X_all, y)

        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]

        if pred == 1:
            detected_clauses.append(clause)
            clause_details.append({
                "clause": clause,
                "risk_level": RISK_CLAUSES.get(clause, 'LOW'),
                "confidence": round(float(prob) * 100, 1),
                "description": CLAUSE_DESCRIPTIONS.get(clause, "")
            })

    risk_score = calculate_risk_score(detected_clauses)

    return {
        "detected_clauses": clause_details,
        "risk_score": risk_score,
        "total_clauses_found": len(detected_clauses),
        "high_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'HIGH'),
        "medium_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'MEDIUM'),
        "low_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'LOW')
    }