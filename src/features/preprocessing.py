from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_preprocessor() -> Pipeline:
    """
    Preprocessing pipeline for linear models.

    Steps:
    1. Median imputation
    2. Standard scaling
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return pipeline


def build_tree_preprocessor() -> Pipeline:
    """
    Preprocessing pipeline for tree-based models.

    Tree models generally do not require scaling,
    so only median imputation is applied.
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    return pipeline