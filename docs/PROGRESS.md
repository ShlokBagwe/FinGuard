# FinGuard AI progress log

Read this first before every task. Update it after every task.

## Done
- Step 1: project setup (git, venv, folders, requirements)
- Step 2: config.yaml + config.py (all paths and settings in one place)
- Step 3: cleaning module (user, card, and transaction data)
  - Fixed `clean_card_data` to keep `expires` and `acct_open_date` as `datetime64`
    and `year_pin_last_changed` as integer year instead of `Period` columns
  - Added `__init__.py` files and `pyproject.toml`, and installed `finguard` in
    editable mode (`pip install -e .`) inside `venv`
  - Verified `scripts/run_cleaning.py` on full data (`users`: (2000, 14),
    `cards`: (6146, 12), `transactions`: (13305915, 13))
- Step 4: merging module (`src/finguard/data/merging.py` and `scripts/run_merging.py`)
  - Verified `master.parquet` output: `(13305915, 38)`, `target` non-null count: `8914963`
- Step 5: feature preprocessing module (`src/finguard/features/preprocessing.py` and
  `scripts/run_features.py`)
  - Removed leaked full-data client aggregates (`customer_avg_amount`,
    `customer_std_amount`, `customer_txn_count`, `amount_vs_avg`, `amount_zscore`)
  - Verified `df_classification.parquet` output: `(8914963, 32)`,
    `transaction_datetime` is `datetime64[us]` with `0` nulls
- Step 6a: velocity module (`src/finguard/features/velocity.py` and
  `scripts/run_velocity.py`)
  - Added 24h rolling `txn_count_last_24h` and `amount_sum_last_24h` deduplicated on
    `(client_id_x, transaction_datetime)` before joining
  - Verified `df_velocity.parquet` output: `(8914963, 34)` with `0` nulls in new columns

## In progress
- (nothing right now)

## Next
- Step 6b: chronological splitting, train-only group stats, and feature
  selection (`03_Feature_Engineering/02_Velocity_Splitting_and_Selection.ipynb`)
- Then: training, evaluation, drift modules
- Then: Streamlit app
- Then: clustering and anomaly detection (notebook first, then modules)

## Notes and decisions
- Model is trained on Kaggle and downloaded as a pickle (whole pipeline + threshold
  + feature list). Library versions must match Kaggle.
- Original FE2 notebook loaded a table with 8,993,227 rows, which is more than the
  8,914,963 labeled rows, so it contained duplicate rows. The module pipeline gives
  8,914,963. Model metrics in the README must be re-measured after the pipeline and
  model are rebuilt.
- Note on FE1 Cell 7 drops: `amount_bins`, `age_group`, `income_group`, `debt_group`
  were EDA-only columns excluded from `master.parquet`.
- Note on FE1 vs. FE2 columns: FE1 Cell 14 creates `is_refund` and drops `amount`.
  FE2 Cell 1/Cell 17 loaded a version of `df_classification_2.parquet` that had
  `amount` instead of `is_refund`.
- 14 exact duplicate rows exist in the features table. They are separate labeled
  transactions whose id column was dropped, so they are kept to match the 8,914,963
  labeled rows. The velocity join adds no duplicates; the old FE2 merge added 78,264.
