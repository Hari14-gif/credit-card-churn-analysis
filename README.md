# Credit Card Analytics Suite

A four-module machine learning project analyzing credit card customer behavior — covering segmentation, default risk, fraud detection, and churn prediction. Built solo in Python (VS Code + Jupyter).

## Overview

Each module tackles a different business problem using a different technique, with a consistent focus on **choosing the right evaluation metric for imbalanced data** rather than defaulting to accuracy.

| Module | Problem | Technique | Key Result |
|---|---|---|---|
| 1. Segmentation | Group customers by behavior | K-Means (k=4, elbow method) + PCA | 4 distinct profiled segments |
| 2. Credit Risk | Predict loan default | Logistic Regression → Random Forest | Recall 24% → 49% |
| 3. Fraud Detection | Flag fraudulent transactions | Logistic Regression → Random Forest | Recall 66% → 76%, Precision 84% → 96% |
| 4. Churn Prediction | Predict customer attrition | Logistic Regression → Random Forest | Recall 52% → 78%, Precision 76% → 92% |

## Module 1 — Customer Segmentation

Used K-Means clustering (k chosen via the elbow method) to segment customers into 4 groups, visualized with PCA for interpretability:
- Heavy Reliance
- Big Spenders
- Modest / Cautious Spenders
- Affluent Underutilized

## Module 2 — Credit Risk / Default Prediction

Built a baseline Logistic Regression model, then improved it with Random Forest (ensemble of decision trees voting). Random Forest traded some precision (69% → 55%) for a large recall gain (24% → 49%) — the right tradeoff here, since missing an actual defaulter is more costly to a bank than a false alarm.

## Module 3 — Fraud Detection

**Dataset:** ULB Credit Card Fraud dataset — 284,807 transactions, only 492 (0.17%) fraudulent.

**The accuracy trap:** a model that predicts "not fraud" for every transaction scores 99.82% accuracy while catching zero fraud. This motivated using recall and precision instead of accuracy throughout.

Used a stratified train/test split to preserve the 0.17% fraud ratio in both sets (critical given how few fraud examples exist). Random Forest improved *both* recall and precision over Logistic Regression — likely because the dataset's features (V1–V28) are already PCA-transformed, making the fraud signal more linearly/non-linearly separable out of the box.

## Module 4 — Churn Prediction

**Dataset:** BankChurners.csv (same source as Module 1) — 16.1% of customers attrited.

**Data leakage caught and fixed:** an initial Logistic Regression model returned a suspiciously perfect 100% accuracy, precision, and recall. Investigation traced this to two leftover columns in the raw dataset (`Naive_Bayes_Classifier_...`) — predictions from someone else's model on the same target, effectively leaking the answer into the features. Dropping these columns before retraining produced realistic, trustworthy results.

Random Forest again improved both recall (52% → 78%) and precision (76% → 92%) over the Logistic Regression baseline.

## Key Takeaways

- **Accuracy is misleading on imbalanced data.** Every module here required recall/precision (or both) instead of accuracy to meaningfully evaluate the model.
- **Stratified splitting matters** when the positive class is rare — a plain random split risks skewing the fraud/default/churn ratio between train and test.
- **Suspiciously perfect scores are a red flag, not a win.** The Module 4 leak was caught specifically because 100% accuracy is essentially never legitimate in a real churn/fraud problem — it's usually a sign the target snuck into the features.
- **Random Forest doesn't always trade precision for recall** — whether it improves both or trades one for the other seems to depend on how separable the signal already is in the raw/engineered features.

## Tech Stack

Python, pandas, scikit-learn (K-Means, PCA, Logistic Regression, Random Forest), Jupyter notebooks.
