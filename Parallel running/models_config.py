# Model sizes from 32x3 to 8192x3
sizes = [8192,32, 6144, 48, 64, 96,4096, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]

models = [[s, s, s] for s in sizes]
model_names = [f"NN{s}x3" for s in sizes]