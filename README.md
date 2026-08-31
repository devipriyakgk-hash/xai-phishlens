# XAI-PhishLens

Phishing website detection using a stacking ensemble (Random Forest + XGBoost + Logistic Regression) explained with **SHAP** (global feature importance) and **LIME** (local, per-prediction reasoning).

This repo backs an IEEE-format conference paper on the same topic. All results below are from real training runs on this dataset — not illustrative placeholders.

## Dataset

10,000 URLs (5,000 phishing / 5,000 legitimate), 16 features covering:
- **Address-bar features**: `Have_IP`, `Have_At`, `URL_Length`, `URL_Depth`, `Redirection`, `https_Domain`, `TinyURL`, `Prefix/Suffix`
- **Domain features**: `DNS_Record`, `Web_Traffic`, `Domain_Age`, `Domain_End`
- **HTML/JS features**: `iFrame`, `Mouse_Over`, `Right_Click`, `Web_Forwards`

Features follow the widely-used UCI Phishing Websites feature schema. Phishing URLs sourced from PhishTank, legitimate URLs from the University of New Brunswick's URL-2016 dataset. Balanced, no missing values.

## Method

1. Three base learners trained independently: Random Forest, XGBoost, Logistic Regression
2. A stacking ensemble combines their out-of-fold predictions via a Logistic Regression meta-learner (5-fold CV stacking)
3. **SHAP** (`TreeExplainer`) explains global feature importance on the strongest base learner (XGBoost)
4. **LIME** explains individual predictions on the full stacked model — including a real misclassified case, for failure analysis

## Results (held-out 20% test set, n=2000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.863 | 0.983 | 0.739 | 0.844 | 0.925 |
| XGBoost | 0.871 | 0.943 | 0.789 | 0.859 | 0.927 |
| Logistic Regression | 0.798 | 0.927 | 0.646 | 0.761 | 0.874 |
| **Stacking Ensemble (ours)** | 0.858 | 0.878 | **0.832** | 0.854 | 0.926 |

**Honest finding**: the stacking ensemble doesn't top every metric — XGBoost alone is competitive on accuracy/F1. What the stack *does* deliver is the best recall (fewest missed phishing sites) with matching ROC-AUC, which matters more than raw accuracy in a security context where a missed phishing site is costlier than a false alarm. This trade-off is worth stating explicitly in the paper rather than only reporting the metric that looks best.

## Explainability findings

**SHAP (global)** — top features driving predictions, ranked by mean |SHAP value|:
1. `URL_Length` — by far the strongest signal
2. `URL_Depth`
3. `Prefix/Suffix`
4. `iFrame`
5. `Domain_Age`

This lines up with known phishing patterns: attackers favor long, obfuscated URLs with excessive subdirectories and suspicious prefixes/suffixes to mimic legitimate domains.

**LIME (local)** — see `results/lime_explanations.md`. Includes a real misclassified phishing URL, where the model leaned on `Prefix/Suffix` and `Have_At` in a way that pushed the prediction toward "legitimate" despite the true label being phishing — useful material for a limitations/error-analysis section.

## Project structure

```
xai-phishlens/
├── data/
│   └── urldata.csv              # raw dataset
├── src/
│   ├── data_prep.py             # loading + train/test split
│   ├── model.py                 # stacking ensemble definition
│   ├── train.py                 # trains all models, saves metrics
│   ├── explain_shap.py          # global SHAP explanations
│   └── explain_lime.py          # local LIME explanations
├── results/                      # generated: metrics, plots, saved model
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
cd src
python train.py          # trains models, saves results/metrics.json + model
python explain_shap.py   # generates SHAP summary plots
python explain_lime.py   # generates LIME local explanations
```

## Next steps

- [ ] Add cross-dataset validation (test on a second phishing dataset, e.g. PhiUSIIL, to check generalization)
- [ ] Extend SHAP analysis to explain the meta-learner directly, not just the strongest base learner
- [ ] Hyperparameter tuning via grid/random search on the stacking ensemble
- [ ] Package as a browser extension or lightweight API for real-time URL scoring

---
*Companion repo for an IEEE-format paper on explainable phishing detection. Built by Devipriya G.*
