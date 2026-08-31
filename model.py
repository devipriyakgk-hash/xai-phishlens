"""
Stacking ensemble for phishing website detection.

Base learners : Random Forest, XGBoost, Logistic Regression
Meta-learner  : Logistic Regression (trained on out-of-fold base predictions)
"""
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_stacking_ensemble(random_state=42):
    base_learners = [
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=random_state, n_jobs=-1
        )),
        ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
    ]

    meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)

    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False,
    )
    return stack
