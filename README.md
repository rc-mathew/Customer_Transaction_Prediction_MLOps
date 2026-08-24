# Customer Transaction Prediction - End-to-End MLOps
[![CI - Tests](https://github.com/rc-mathew/Customer_Transaction_Prediction_MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/rc-mathew/Customer_Transaction_Prediction_MLOps/actions/workflows/ci.yml)
An end-to-end machine learning project for customer transaction prediction, covering model development, evaluation, deployment, CI/CD, containerization, monitoring, subgroup stability analysis, calibration, and cloud deployment.

## Implemented MLOps Components

- Data preprocessing and feature engineering
- Model training and evaluation
- ROC-AUC and classification metrics
- Probability calibration
- Subgroup stability analysis
- REST API deployment
- Automated testing
- Docker containerization
- CI/CD with GitHub Actions
- Kubernetes deployment
- Data and model drift monitoring
- Cloud deployment
## Model Evaluation Results

The production ML pipeline was evaluated using stratified validation, probability calibration, and decision-threshold optimization.
## REST API Deployment

The trained HistGradientBoosting pipeline is exposed through a FastAPI inference service.

### API Endpoints

- `GET /` — API information
- `GET /health` — model health check
- `POST /predict` — customer transaction prediction

The API loads the serialized production model from:

```text
models/customer_transaction_model.joblib

### Decision Threshold Optimization

A fixed classification threshold of 0.50 is not necessarily optimal for an imbalanced classification problem. The decision threshold was therefore optimized on the validation data by maximizing the F1 score.

The selected threshold was:

**0.26**

Validation performance at the selected threshold:

| Metric | Score |
|---|---:|
| Precision | 0.511 |
| Recall | 0.560 |
| F1 Score | 0.534 |
| Balanced Accuracy | 0.750 |

### Probability Calibration using validation data

Predicted probabilities were calibrated using **Isotonic Regression**.

Calibration was performed using validation data rather than the untouched test set to reduce the risk of test-set leakage.

### Production Evaluation Design

- Stratified train/validation/test splitting
- Probability calibration using validation data
- Validation-based decision threshold optimization
- Subgroup stability analysis
- Data drift monitoring
- Automated tests for preprocessing, threshold logic, and drift detection
- Untouched test set retained for final evaluation

