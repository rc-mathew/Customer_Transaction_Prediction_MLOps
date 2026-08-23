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