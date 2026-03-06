# Model sizes from 32x3 to 8192x3
sizes = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]

models = [[s, s, s] for s in sizes]
model_names = [f"NN{s}x3" for s in sizes]