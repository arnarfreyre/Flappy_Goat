# Model sizes from 32x3 to 8192x3
#sizes = [50,50,50]
sizes = [64,128,64,128]
models = [[s, s, s] for s in sizes]
model_names = [f"NN{s}x3" for s in sizes]