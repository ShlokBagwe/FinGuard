---
name: notebook-to-module
description: Step-by-step workflow for moving a section of a Jupyter notebook into a function in the finguard package. Use when the user asks to convert, move or port notebook code into a module.
---

# Notebook to module

## Goal
Move working notebook logic into a module function WITHOUT changing the result.

## Steps
1. Read only the notebook cells the user names. Do not read the whole notebook.
2. Copy the logic exactly. Do not improve, rename columns, or change numbers.
3. Wrap it in the function name, inputs and outputs the user gave.
4. Remove everything that is not logic: plots, display(), shape prints, EDA checks.
5. Replace hardcoded paths and settings with values from the config. Keep column
   names that belong to the cleaning logic inside the function.
6. Write a short run check (in a scripts/ file or at the end of your reply) that runs
   the function and prints shape and columns, so the user can compare them with the
   notebook output.
7. Tell the user clearly about any difference you noticed or any cell you skipped.

## Leakage rule (important for this project)
Anything learned from data (averages, fraud rates per MCC/city/state, customer
stats) must be learned on the TRAIN split only and then applied to the test split.
If the notebook did this in two steps, keep two functions: one that learns from
train, one that applies to any data.

## Memory rule
Raw data is about 13M rows and the laptop has 16GB RAM. Save in-between files as
parquet, not CSV, and load only the columns needed.
