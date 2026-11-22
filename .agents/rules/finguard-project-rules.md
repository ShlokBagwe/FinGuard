# FinGuard AI project rules

## Before every task
1. Read docs/PROGRESS.md to see what is done and what is next.
2. Check the available skills. For code under src/, scripts/, app/ or tests/, use
   `finguard-code-conventions`. When turning notebook code into module code, also use
   `notebook-to-module`.

## What this project is
Fraud detection on 13.3M card transactions (XGBoost). We are turning notebooks into a
clean modular Python package called `finguard` inside src/. Later we add a Streamlit
app, customer segmentation, anomaly detection and a risk layer.

## Who decides what
- The user and their reviewer decide the structure, file names, function names and
  what goes in and out of each function.
- You write the code for ONE small task at a time, exactly as asked.
- If a task needs a decision that was not given (new folder, new library, renaming,
  changing the structure), STOP and ask. Do not decide on your own.

## Always
- Keep code simple, plain names, step by step. No clever tricks.
- No extra helper functions, classes, asserts, or abstractions unless asked.
- Never change model logic, hyperparameters, or results from the notebooks.
- Never edit or delete anything in data/raw/.
- Never hardcode paths such as /kaggle/... or C:/...; read paths from the config.
- Do not add a new library to requirements.txt without asking.

## After every task
1. Update docs/PROGRESS.md (what was done, what is next).
2. End with a short summary in simple words: which files changed, what each function
   does, and anything you were unsure about.
