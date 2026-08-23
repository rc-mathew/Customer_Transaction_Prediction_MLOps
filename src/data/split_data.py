import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def split_dataset(df: pd.DataFrame):
    """
    Split the dataset into train, validation and test sets
    using stratification on the target variable.

    Returns
    -------
    tuple
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        id_train, id_valid, id_test
    """

    X = df.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ]
    )

    y = df[TARGET_COLUMN]

    ids = df[ID_COLUMN]

    # -----------------------------------------------------
    # FIRST SPLIT: TRAIN+VALIDATION VS TEST
    # -----------------------------------------------------

    (
        X_temp,
        X_test,
        y_temp,
        y_test,
        id_temp,
        id_test,
    ) = train_test_split(
        X,
        y,
        ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -----------------------------------------------------
    # SECOND SPLIT: TRAIN VS VALIDATION
    # -----------------------------------------------------

    validation_fraction = (
        VALIDATION_SIZE / (1 - TEST_SIZE)
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        id_train,
        id_valid,
    ) = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Train shape: %s",
        X_train.shape,
    )

    logger.info(
        "Validation shape: %s",
        X_valid.shape,
    )

    logger.info(
        "Test shape: %s",
        X_test.shape,
    )

    print("\nTARGET DISTRIBUTIONS")
    print("=" * 60)

    print(
        "Train:",
        y_train.value_counts(normalize=True)
        .sort_index()
        .to_dict(),
    )

    print(
        "Validation:",
        y_valid.value_counts(normalize=True)
        .sort_index()
        .to_dict(),
    )

    print(
        "Test:",
        y_test.value_counts(normalize=True)
        .sort_index()
        .to_dict(),
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        id_train,
        id_valid,
        id_test,
    )


if __name__ == "__main__":

    from src.data.load_data import load_training_data
    from src.data.validate_data import validate_dataset

    data = load_training_data()

    validate_dataset(data)

    split_dataset(data)