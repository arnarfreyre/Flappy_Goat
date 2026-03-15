# Model sizes from 32x3 to 8192x3
#sizes = [50,50,50]
sizes = [700,600,450]
models = [[s, s, s] for s in sizes]
model_names = [f"NN{s}x3" for s in sizes]

# Training / convergence settings
MAX_PIPES = 100000
CONVERGENCE_STREAK = 3
RUNS_PER_MODEL = 2
TOTAL_WORKERS = 2