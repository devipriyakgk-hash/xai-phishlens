"""
Local explainability using LIME on the full stacking ensemble.
Explains a handful of individual predictions (not global patterns) —
this is LIME's actual purpose: "why did the model flag THIS specific URL?"
"""
import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from data_prep import FEATURE_COLS


def main():
    stack = joblib.load('results/stacking_model.joblib')
    X_test = pd.read_csv('results/X_test.csv')
    y_test = pd.read_csv('results/y_test.csv')['Label']

    explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=FEATURE_COLS,
        class_names=['Legitimate', 'Phishing'],
        discretize_continuous=True,
        mode='classification'
    )

    # Explain 3 sample predictions: a true positive, a true negative,
    # and a misclassification if one exists (most instructive case).
    preds = stack.predict(X_test)
    correct_phish = X_test[(preds == 1) & (y_test.values == 1)].index[:1]
    correct_legit = X_test[(preds == 0) & (y_test.values == 0)].index[:1]
    wrong = X_test[preds != y_test.values].index[:1]

    sample_idxs = list(correct_phish) + list(correct_legit) + list(wrong)
    labels = ['correctly_flagged_phishing', 'correctly_flagged_legitimate', 'misclassified_example']

    report_lines = []
    for idx, label in zip(sample_idxs, labels):
        exp = explainer.explain_instance(
            X_test.loc[idx].values, stack.predict_proba, num_features=6
        )
        report_lines.append(f"\n## {label} (row {idx})")
        report_lines.append(f"True label: {'Phishing' if y_test.loc[idx]==1 else 'Legitimate'}, "
                             f"Predicted: {'Phishing' if preds[idx]==1 else 'Legitimate'}")
        for feature, weight in exp.as_list():
            report_lines.append(f"- {feature}: {weight:+.4f}")
        exp.save_to_file(f'results/lime_{label}.html')

    with open('results/lime_explanations.md', 'w') as f:
        f.write("# LIME Local Explanations\n" + "\n".join(report_lines))

    print("\n".join(report_lines))
    print("\nSaved: results/lime_explanations.md and per-instance HTML reports")


if __name__ == '__main__':
    main()
