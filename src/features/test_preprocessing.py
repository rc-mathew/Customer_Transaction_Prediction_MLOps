from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
)


def main():

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

    linear_preprocessor = (
        build_linear_preprocessor()
    )

    tree_preprocessor = (
        build_tree_preprocessor()
    )

    X_train_linear = (
        linear_preprocessor.fit_transform(
            X_train
        )
    )

    X_train_tree = (
        tree_preprocessor.fit_transform(
            X_train
        )
    )

    print("\nPREPROCESSING TEST")
    print("=" * 60)

    print(
        "Original shape:",
        X_train.shape
    )

    print(
        "Linear processed shape:",
        X_train_linear.shape
    )

    print(
        "Tree processed shape:",
        X_train_tree.shape
    )

    print(
        "\nPreprocessing completed successfully."
    )


if __name__ == "__main__":
    main()