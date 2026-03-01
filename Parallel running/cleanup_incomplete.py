"""Remove CSV run files that are not listed as complete in overview.csv."""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
OVERVIEW = os.path.join(DATA_DIR, 'overview.csv')

# Load completed filenames from overview.csv
completed = set()
if os.path.exists(OVERVIEW):
    with open(OVERVIEW) as f:
        for row in csv.DictReader(f):
            completed.add((row['model'], row['csv_file']))

# Walk each model folder and remove non-completed CSVs
removed = []
for model_dir in sorted(os.listdir(DATA_DIR)):
    model_path = os.path.join(DATA_DIR, model_dir)
    if not os.path.isdir(model_path) or model_dir == '__pycache__':
        continue
    for fname in os.listdir(model_path):
        if not fname.endswith('.csv'):
            continue
        if (model_dir, fname) not in completed:
            full = os.path.join(model_path, fname)
            os.remove(full)
            removed.append(f"  {model_dir}/{fname}")

if removed:
    print(f"Removed {len(removed)} incomplete run(s):")
    for r in removed:
        print(r)
else:
    print("No incomplete runs found.")
