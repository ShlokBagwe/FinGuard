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

## In progress
- (nothing right now)

## Next
- Step 5: features module (`03_Feature_Engineering/01_Feature_Engineering_Preprocessing.ipynb`)
- Then: splitting, training, evaluation, drift modules
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
