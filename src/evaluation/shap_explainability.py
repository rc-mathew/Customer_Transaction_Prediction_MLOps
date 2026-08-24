from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data.load_data import load_training_data


MODEL_PATH = Path("models/customer_transaction_model.joblib")
OUTPUT_DIR = Path("reports/shap")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the 200 Santander-style feature columns."""
    return [f"var_{i}" for i in range(200) if f"var_{i}" in df.columns]


def load_model():
    """Load the serialized production model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def main():
    print("\nSHAP MODEL EXPLAINABILITY")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()
    feature_columns = get_feature_columns(df)

    if len(feature_columns) != 200:
        raise ValueError(
            f"Expected 200 feature columns, found {len(feature_columns)}."
        )

    X = df[feature_columns]

    model = load_model()

    # Keep the explanation lightweight enough for a laptop.
    background = shap.sample(
        X,
        50,
        random_state=42,
    )

    explanation_data = shap.sample(
        X,
        100,
        random_state=43,
    )

    def predict_positive_class(data):
        data_df = pd.DataFrame(
            data,
            columns=feature_columns,
        )

        return model.predict_proba(data_df)[:, 1]

    print("Creating SHAP explainer...")

    explainer = shap.KernelExplainer(
        predict_positive_class,
        background,
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        explanation_data,
        nsamples=100,
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    print("Creating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_columns,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    summary_path = OUTPUT_DIR / "shap_summary.png"

    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    mean_abs_shap = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_shap": abs(shap_values).mean(axis=0),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_path = (
        OUTPUT_DIR / "shap_feature_importance.csv"
    )

    mean_abs_shap.to_csv(
        importance_path,
        index=False,
    )

    print("\nTop 20 SHAP Features")
    print("-" * 60)
    print(
        mean_abs_shap.head(20).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {importance_path}")


if __name__ == "__main__":
    main()