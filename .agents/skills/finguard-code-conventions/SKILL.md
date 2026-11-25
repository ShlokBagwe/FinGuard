---
name: finguard-code-conventions
description: Coding conventions and folder map for the FinGuard AI repo. Use whenever creating or editing any file under src/finguard, scripts, app or tests.
---

# FinGuard code conventions

## Goal
Keep every module in the repo consistent, simple and easy to review.

## Folder map
- src/finguard/data/        cleaning.py, merging.py, splitting.py
- src/finguard/features/    feature building (time, ratio, velocity, group stats)
- src/finguard/models/      train.py, evaluate.py, threshold.py, predict.py
- src/finguard/monitoring/  drift.py (PSI)
- scripts/                  small files that run a pipeline step from the terminal
- app/                      Streamlit app
- models/                   saved model files
- data/raw, interim, processed   data files (never committed to git)

## How to write a function
1. One function does one job, named like a verb: clean_user_data, add_time_features.
2. It takes data in as arguments and returns data out. No global variables.
3. Do not change the input DataFrame in place; work on a copy and return it.
4. Add a short docstring: what it does, what it takes, what it returns (2-3 lines).
5. Paths and model settings come from the config. Column names that are part of
   the cleaning logic stay inside the function.
6. Use plain print() for simple progress messages. No plots or display() in modules.
7. Use pathlib for file paths so they work on Windows.
8. Comments explain why, not what. Keep them short and professional. Never number
   comments (Step 1, Step 2) and never narrate obvious lines.
9. Use private functions whenever necessary.

## Do not
- Do not write classes unless asked.
- Do not add error handling, logging, type-heavy code or asserts unless asked.
- Do not put code that runs on import. Running code belongs in scripts/.
