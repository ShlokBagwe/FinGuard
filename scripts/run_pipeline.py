from pathlib import Path
import subprocess
import sys
import time

STAGES = [
    "run_cleaning",
    "run_merging",
    "run_features",
    "run_velocity",
    "run_splitting",
    "train_model",
    "run_drift",
]

SCRIPTS_DIR = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    start_stage = sys.argv[1]
    if start_stage not in STAGES:
        print("Unknown stage:", start_stage)
        print("Valid stages:", ", ".join(STAGES))
        sys.exit(1)
    stages_to_run = STAGES[STAGES.index(start_stage) :]
else:
    stages_to_run = STAGES

for stage in stages_to_run:
    script_path = SCRIPTS_DIR / f"{stage}.py"
    print(f"Starting: {stage}")
    start_time = time.perf_counter()

    # Run each stage in a fresh subprocess so memory is freed between steps.
    result = subprocess.run([sys.executable, str(script_path)], cwd=SCRIPTS_DIR.parent)
    elapsed = time.perf_counter() - start_time

    if result.returncode != 0:
        print(f"Failed: {stage} (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"Finished: {stage} ({elapsed:.2f}s)")
