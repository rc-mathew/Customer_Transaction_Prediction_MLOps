# Dataset

The raw customer transaction dataset is intentionally excluded from Git version control.

Place the training dataset at:

data/raw/train.csv

Expected schema:

- `ID_code`: customer identifier
- `target`: binary prediction target
- `var_0` to `var_199`: 200 anonymized numerical predictors

Target interpretation:

- `0`: customer will not make the transaction
- `1`: customer will make the transaction

The raw dataset is excluded through `.gitignore` to avoid committing large source data files.
---

## Model Development and Evaluation

The machine learning workflow was developed first as an end-to-end experimentation notebook and then refactored into reusable production-oriented modules.

### Development Notebook

The complete exploratory and model-development workflow is available here:

[`notebooks/customer_transaction_prediction_end_to_end.ipynb`](notebooks/customer_transaction_prediction_end_to_end.ipynb)

The notebook covers:

- Data quality assessment
- Exploratory data analysis
- Class imbalance analysis
- Feature preprocessing
- Baseline modelling
- Logistic Regression
- HistGradientBoosting
- ROC-AUC and PR-AUC evaluation
- Stratified cross-validation
- Hyperparameter tuning
- Probability calibration
- Decision-threshold analysis
- Model interpretation and validation

---

## Model Performance

The dataset is highly imbalanced, so model selection is not based on accuracy alone.

Primary evaluation metrics include:

- ROC-AUC
- PR-AUC / Average Precision
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Brier Score for probability calibration

### Baseline vs Gradient Boosting

| Model | ROC-AUC | Notes |
|---|---:|---|
| Logistic Regression | ~0.854 | Interpretable baseline |
| HistGradientBoosting | ~0.880 | Stronger nonlinear benchmark |
| Tuned HistGradientBoosting | See tuning results | Selected through stratified CV and hyperparameter search |

The gradient-boosting model produced stronger ranking performance than the linear baseline and was therefore selected for further production hardening.

Detailed tuning results are available in:

[`reports/metrics/hist_gradient_tuning_results.json`](reports/metrics/hist_gradient_tuning_results.json)

---

## Cross-Validation

Model performance was evaluated using stratified cross-validation rather than relying on a single train-validation split.

The cross-validation stage compares:

- Mean ROC-AUC
- ROC-AUC standard deviation
- Mean PR-AUC
- PR-AUC standard deviation
- Accuracy
- Balanced accuracy
- F1 score

Implementation:

[`src/evaluation/cross_validate_models.py`](src/evaluation/cross_validate_models.py)

This provides evidence that model performance is stable across multiple folds.

---

## Hyperparameter Optimization

`RandomizedSearchCV` was used to optimize the HistGradientBoosting model while controlling computational cost.

Parameters explored include:

- Learning rate
- Number of boosting iterations
- Maximum leaf nodes
- Minimum samples per leaf
- L2 regularization

Implementation:

[`src/models/tune_hist_gradient_boosting.py`](src/models/tune_hist_gradient_boosting.py)

The search objective is ROC-AUC because the target distribution is highly imbalanced.

---

## Probability Calibration

For financial ML applications, ranking customers correctly is not sufficient. Predicted probabilities should also represent meaningful risk estimates.

Three probability configurations were evaluated:

1. Uncalibrated model
2. Sigmoid calibration
3. Isotonic calibration

Calibration quality is evaluated using:

- Brier Score
- Log Loss
- ROC-AUC
- PR-AUC

The calibration method with the lowest validation Brier Score is selected.

Implementation:

[`src/evaluation/calibration.py`](src/evaluation/calibration.py)

Results:

[`reports/metrics/calibration_results.csv`](reports/metrics/calibration_results.csv)

### Calibration Curve

![Probability Calibration Curve](reports/figures/calibration_curve.png)

---

## Decision Threshold Optimization

A fixed probability threshold of `0.50` is not automatically optimal for an imbalanced classification problem.

The project therefore evaluates multiple decision thresholds and compares:

- Precision
- Recall
- F1 Score
- Balanced Accuracy

The validation analysis selected a threshold around:

```text
0.26
The threshold was selected by maximizing validation F1 score.

At the selected threshold, validation performance was approximately:

| Metric | Score |
|---|---:|
| Precision | 0.511 |
| Recall | 0.560 |
| F1 Score | 0.534 |
| Balanced Accuracy | 0.750 |

Probability calibration method: **Isotonic**