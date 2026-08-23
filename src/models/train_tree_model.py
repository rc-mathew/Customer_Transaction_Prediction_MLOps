from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import build_tree_preprocessor


def main():

    print("\nLoading data...")

    df = load_training_data()

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        id_train,
        id_valid,
        id_test,
    ) = split_dataset(df)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining HistGradientBoosting model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    print("\nHISTGRADIENTBOOSTING VALIDATION RESULTS")
    print("=" * 55)

    print(
        f"Accuracy:          "
        f"{accuracy_score(y_valid, predictions):.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(y_valid, predictions):.4f}"
    )

    print(
        f"Precision:         "
        f"{precision_score(y_valid, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:            "
        f"{recall_score(y_valid, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 Score:          "
        f"{f1_score(y_valid, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{roc_auc_score(y_valid, probabilities):.4f}"
    )

    print(
        f"PR-AUC:            "
        f"{average_precision_score(y_valid, probabilities):.4f}"
    )


if __name__ == "__main__":
    main()