"""Move incomplete CSV run files to Data/ToDelete/ instead of deleting them.

A run is considered complete if its last 3 epoch_test_pipes values are all 100000.
Completed runs get their trailing zero-rows forward-filled from the last real epoch.
"""

import os
import shutil
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
TO_DELETE_DIR = os.path.join(DATA_DIR, 'ToDelete')
FILL_COLS = ['epoch_pipes', 'epoch_loss_clip', 'epoch_loss_val', 'epoch_loss_ent', 'epoch_loss_tot']


def is_completed(df):
    """Check if run ends with 3 consecutive epoch_test_pipes == 100000."""
    if len(df) < 3 or 'epoch_test_pipes' not in df.columns:
        return False
    last3 = df['epoch_test_pipes'].iloc[-3:].tolist()
    return all(v == 100000 for v in last3)


# Walk each model folder
moved = []
fixed_files = []

for model_dir in sorted(os.listdir(DATA_DIR)):
    model_path = os.path.join(DATA_DIR, model_dir)
    if not os.path.isdir(model_path) or model_dir in ('__pycache__', 'ToDelete'):
        continue
    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(model_path, fname)
        df = pd.read_csv(fpath)

        if not is_completed(df):
            # Move incomplete run to ToDelete
            dest_dir = os.path.join(TO_DELETE_DIR, model_dir)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(fpath, os.path.join(dest_dir, fname))
            moved.append(f"  {model_dir}/{fname}")
            continue

        # Forward-fill only TRAILING zero rows in completed runs
        # Walk backwards from end to find trailing rows where epoch_pipes==0
        n = len(df)
        trailing_start = n
        for i in range(n - 1, -1, -1):
            if df.loc[i, 'epoch_pipes'] == 0 and df.loc[i, 'epoch_test_pipes'] == 100000:
                trailing_start = i
            else:
                break

        if trailing_start >= n:
            continue

        last_real_idx = trailing_start - 1
        if last_real_idx < 0:
            continue

        num_filled = n - trailing_start
        for col in FILL_COLS:
            if col in df.columns:
                df.loc[trailing_start:, col] = df.loc[last_real_idx, col]

        df.to_csv(fpath, index=False)
        fixed_files.append(f"  {model_dir}/{fname} ({num_filled} rows)")

if moved:
    print(f"Moved {len(moved)} incomplete run(s) to {TO_DELETE_DIR}:")
    for m in moved:
        print(m)
else:
    print("No incomplete runs found.")

if fixed_files:
    print(f"\nForward-filled {len(fixed_files)} file(s):")
    for f in fixed_files:
        print(f)
else:
    print("\nNo forward-fill fixes needed.")
