"""
Global explainability using SHAP on the XGBoost base learner
(SHAP's TreeExplainer is used here since it's exact and fast for tree models;
the stacking meta-learner's decision is a linear combination of base-learner
outputs, so explaining the strongest base learner gives an interpretable
proxy for what drives the ensemble).
"""
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from data_prep import FEATURE_COLS


def main():
    stack = joblib.load('results/stacking_model.joblib')
    xgb_model = dict(stack.named_estimators_)['xgb']

    X_test = pd.read_csv('results/X_test.csv')

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png', dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS,
                       plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('results/shap_importance_bar.png', dpi=150)
    plt.close()

    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv('results/shap_feature_ranking.csv', header=['mean_abs_shap'])

    print("Top 5 features by mean |SHAP value|:")
    print(mean_abs_shap.head(5))
    print("\nSaved: results/shap_summary.png, results/shap_importance_bar.png, "
          "results/shap_feature_ranking.csv")


if __name__ == '__main__':
    main()
