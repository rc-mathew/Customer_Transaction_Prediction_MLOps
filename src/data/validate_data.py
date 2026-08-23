import logging

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_FEATURE_COUNT,
    ID_COLUMN,
    TARGET_COLUMN,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate structure and quality of the customer
    transaction dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict
        Data validation results.

    Raises
    ------
    ValueError
        If critical schema requirements are violated.
    """

    validation_results = {}

    # -----------------------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------------------

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_required = required_columns - set(df.columns)

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    # -----------------------------------------------------
    # FEATURE COLUMNS
    # -----------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [ID_COLUMN, TARGET_COLUMN]
    ]

    validation_results["rows"] = len(df)
    validation_results["columns"] = len(df.columns)
    validation_results["feature_count"] = len(feature_columns)

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected number of features. "
            f"Expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(feature_columns)}."
        )

    # -----------------------------------------------------
    # TARGET VALIDATION
    # -----------------------------------------------------

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Target column must contain only 0 and 1. "
            f"Found values: {target_values}"
        )

    validation_results["target_classes"] = sorted(
        target_values
    )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    validation_results["missing_values"] = int(
        df.isna().sum().sum()
    )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    validation_results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    validation_results["duplicate_ids"] = int(
        df[ID_COLUMN].duplicated().sum()
    )

    # -----------------------------------------------------
    # INFINITE VALUES
    # -----------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    validation_results["infinite_values"] = int(
        np.isinf(numeric_df).sum().sum()
    )

    # -----------------------------------------------------
    # FEATURE TYPES
    # -----------------------------------------------------

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    validation_results[
        "non_numeric_features"
    ] = non_numeric_features

    # -----------------------------------------------------
    # TARGET DISTRIBUTION
    # -----------------------------------------------------

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )

    validation_results[
        "target_distribution"
    ] = target_distribution

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    logger.info("Dataset validation completed.")

    print("\nDATA VALIDATION REPORT")
    print("=" * 60)

    for key, value in validation_results.items():
        print(f"{key:25s}: {value}")

    return validation_results


if __name__ == "__main__":

    from src.data.load_data import load_training_data

    data = load_training_data()

    validate_dataset(data)