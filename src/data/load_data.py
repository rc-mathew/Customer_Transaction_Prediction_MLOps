import logging
from pathlib import Path

import pandas as pd

from src.config import TRAIN_DATA_PATH


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_training_data(
    path: Path = TRAIN_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the customer transaction training dataset.

    Parameters
    ----------
    path : Path
        Location of the training CSV file.

    Returns
    -------
    pd.DataFrame
        Customer transaction dataset.

    Raises
    ------
    FileNotFoundError
        If the training file cannot be found.

    ValueError
        If the dataset is empty.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at: {path}"
        )

    logger.info("Loading training dataset from %s", path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    logger.info(
        "Dataset loaded successfully: %s rows, %s columns",
        f"{df.shape[0]:,}",
        df.shape[1],
    )

    return df


if __name__ == "__main__":
    data = load_training_data()

    print("\nDataset Preview")
    print("-" * 60)
    print(data.head())

    print("\nDataset Shape")
    print("-" * 60)
    print(data.shape)