"""
Trains base learners individually (for comparison) and the stacking
ensemble, then reports real evaluation metrics on a held-out test set.
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

from data_prep import get_train_test
from model import build_stacking_ensemble


def evaluate(name, model, X_test, y_test, results):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'accuracy': round(accuracy_score(y_test, preds), 4),
        'precision': round(precision_score(y_test, preds), 4),
        'recall': round(recall_score(y_test, preds), 4),
        'f1': round(f1_score(y_test, preds), 4),
        'roc_auc': round(roc_auc_score(y_test, probs), 4),
    }
    print(f"{name:22s} | " + " | ".join(f"{k}={v}" for k, v in results[name].items()))


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    results = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate('RandomForest', rf, X_test, y_test, results)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    evaluate('XGBoost', xgb, X_test, y_test, results)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate('LogisticRegression', lr, X_test, y_test, results)

    stack = build_stacking_ensemble()
    stack.fit(X_train, y_train)
    evaluate('SHAP-LIME-Stack (ours)', stack, X_test, y_test, results)

    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    joblib.dump(stack, 'results/stacking_model.joblib')
    X_test.to_csv('results/X_test.csv', index=False)
    y_test.to_csv('results/y_test.csv', index=False)

    print("\nSaved: results/metrics.json, results/stacking_model.joblib")


if __name__ == '__main__':
    main()
