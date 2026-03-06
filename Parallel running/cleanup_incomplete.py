"""Move incomplete CSV run files to Data/ToDelete/ instead of deleting them."""

import csv
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
OVERVIEW = os.path.join(DATA_DIR, 'overview.csv')
TO_DELETE_DIR = os.path.join(DATA_DIR, 'ToDelete')

# Load completed filenames from overview.csv
completed = set()
if os.path.exists(OVERVIEW):
    with open(OVERVIEW) as f:
        for row in csv.DictReader(f):
            completed.add((row['model'], row['csv_file']))

# Walk each model folder and move non-completed CSVs to ToDelete
moved = []
for model_dir in sorted(os.listdir(DATA_DIR)):
    model_path = os.path.join(DATA_DIR, model_dir)
    if not os.path.isdir(model_path) or model_dir in ('__pycache__', 'ToDelete'):
        continue
    for fname in os.listdir(model_path):
        if not fname.endswith('.csv'):
            continue
        if (model_dir, fname) not in completed:
            dest_dir = os.path.join(TO_DELETE_DIR, model_dir)
            os.makedirs(dest_dir, exist_ok=True)
            src = os.path.join(model_path, fname)
            shutil.move(src, os.path.join(dest_dir, fname))
            moved.append(f"  {model_dir}/{fname}")

if moved:
    print(f"Moved {len(moved)} incomplete run(s) to {TO_DELETE_DIR}:")
    for m in moved:
        print(m)
else:
    print("No incomplete runs found.")
