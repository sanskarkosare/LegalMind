import joblib
import numpy as np
import os

MODELS_DIR = "models"
TFIDF_PATH = os.path.join(MODELS_DIR, "legal_tfidf.pkl")

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

CLAUSE_TO_FILE = {
    'Non-Compete': 'non-compete_model.pkl',
    'Exclusivity': 'exclusivity_model.pkl',
    'Termination For Convenience': 'termination_for_convenience_model.pkl',
    'Change Of Control': 'change_of_control_model.pkl',
    'Anti-Assignment': 'anti-assignment_model.pkl',
    'Ip Ownership Assignment': 'ip_ownership_assignment_model.pkl',
    'License Grant': 'license_grant_model.pkl',
    'Cap On Liability': 'cap_on_liability_model.pkl',
    'Uncapped Liability': 'uncapped_liability_model.pkl',
    'Audit Rights': 'audit_rights_model.pkl',
    'Insurance': 'insurance_model.pkl',
    'Liquidated Damages': 'liquidated_damages_model.pkl',
    'Warranty Duration': 'warranty_duration_model.pkl',
    'Non-Disparagement': 'non-disparagement_model.pkl',
    'No-Solicit Of Employees': 'no-solicit_of_employees_model.pkl'
}

def load_tfidf():
    return joblib.load(TFIDF_PATH)

def load_clause_model(clause):
    filename = CLAUSE_TO_FILE.get(clause)
    if not filename:
        return None
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def calculate_risk_score(detected_clauses):
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
    elif normalized >= 35:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"

    return {"score": round(normalized, 1), "level": level, "raw_score": score}

def analyze_contract(text):
    try:
        tfidf = load_tfidf()
    except Exception as e:
        return {"error": f"TF-IDF model not found: {str(e)}"}

    X_new = tfidf.transform([text[:3000]])

    detected_clauses = []
    clause_details = []

    for clause in RISK_CLAUSES.keys():
        model = load_clause_model(clause)
        if model is None:
            continue
        try:
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
        except:
            continue

    risk_score = calculate_risk_score(detected_clauses)

    return {
        "detected_clauses": clause_details,
        "risk_score": risk_score,
        "total_clauses_found": len(detected_clauses),
        "high_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'HIGH'),
        "medium_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'MEDIUM'),
        "low_risk_count": sum(1 for c in clause_details if c['risk_level'] == 'LOW')
    }
