import random

def get_random_layers():
    input_layer = 8
    total_layers = random.randint(1, 16)
    layer_specs = []
    layer_specs.append([input_layer, 2 ** (3 + random.randint(1, 8))])

    for i in range(total_layers - 1):
        layer_specs.append([layer_specs[i][1], int(layer_specs[i][1]*2 ** (random.randint(-2, 2)))])

    total_size = 0
    for i in layer_specs:
        size = i[0]*i[1] + i[1]
        total_size += size
        print(size)

    print(total_size)
    print(f"Total layers: {total_layers}, layer specs: {layer_specs}")

    return total_layers, layer_specs


get_random_layers()