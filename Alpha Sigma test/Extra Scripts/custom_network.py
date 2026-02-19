import numpy as np
import os

np.random.seed(42)

""" --------------------------------------------- classes --------------------------------------------------"""

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def mfunc2(a=100000,b=1,c=0.5,d=0,x=None):
    """
    :param a: scales precision
    :param b: moves function up and down
    :param c: b*c = dist(min,max)
    :param d: x location of spike
    :param x: w'*x
    :return: talpha function
    """
    z = a*x/2-d*a

    func = np.tanh(z)-1
    ret_func = (c*(func)+b)
    return ret_func


class Layer:
    def __init__(self, neurons, idx, activation=None):
        self.neurons = neurons
        self.idx = idx
        self.activation = activation
        self.c = None
        self.d = None


class Network:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []

    def add_layer(self, neurons, activation=None):
        idx = len(self.layers)
        self.layers.append(Layer(neurons, idx, activation))

    def add_linear(self, neurons, activation=None):
        self.add_layer(neurons, activation)

    def init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i].neurons, self.layers[i + 1].neurons) * 0.1
            b = np.zeros((1, self.layers[i + 1].neurons))
            self.weights.append(W)
            self.biases.append(b)
            dest = self.layers[i + 1]
            if dest.activation is mfunc2:
                dest.c = np.full((1, dest.neurons), 0.5)
                dest.d = np.zeros((1, dest.neurons))

    def forward(self, x, cache=False):
        a = x
        if cache:
            self._activations = [a]
            self._pre_activations = []
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            if cache:
                self._pre_activations.append(z)
            layer = self.layers[i + 1]
            if layer.activation is mfunc2:
                a = mfunc2(a=100000, b=1, c=layer.c, d=layer.d, x=z)
            elif layer.activation is not None:
                a = layer.activation(z)
            else:
                a = z
            if cache:
                self._activations.append(a)
        return a

    def loss(self, x, y_true):
        return (0.5 * (self.forward(x) - y_true) ** 2).item()

    def compute_gradients(self, x, y_true):
        a_const = 100000
        output = self.forward(x, cache=True)
        dL_da = output - y_true

        self.w_grads = [None] * len(self.weights)
        self.b_grads = [None] * len(self.biases)
        self.c_grads = [None] * len(self.weights)
        self.d_grads = [None] * len(self.weights)

        for i in range(len(self.weights) - 1, -1, -1):
            layer = self.layers[i + 1]
            z = self._pre_activations[i]

            if layer.activation is mfunc2:
                u = a_const * z / 4 - layer.d * a_const / 2
                sech2 = 1 - np.tanh(u) ** 2
                func_val = np.tanh(u) - 1

                self.c_grads[i] = dL_da * func_val
                self.d_grads[i] = dL_da * layer.c * sech2 * (-a_const / 2)
                dL_dz = dL_da * layer.c * sech2 * (a_const / 4)
            elif layer.activation is relu:
                dL_dz = dL_da * (z > 0).astype(float)
            elif layer.activation is tanh:
                dL_dz = dL_da * (1 - np.tanh(z) ** 2)
            else:
                dL_dz = dL_da

            a_prev = self._activations[i]
            self.w_grads[i] = a_prev.T @ dL_dz
            self.b_grads[i] = dL_dz

            if i > 0:
                dL_da = dL_dz @ self.weights[i].T

    def update_weights(self, lr=0.001, clip=1.0):
        for i in range(len(self.weights)):
            self.w_grads[i] = np.clip(self.w_grads[i], -clip, clip)
            self.b_grads[i] = np.clip(self.b_grads[i], -clip, clip)
            self.weights[i] -= lr * self.w_grads[i]
            self.biases[i] -= lr * self.b_grads[i]
            if self.c_grads[i] is not None:
                self.c_grads[i] = np.clip(self.c_grads[i], -clip, clip)
                self.d_grads[i] = np.clip(self.d_grads[i], -clip, clip)
                self.layers[i + 1].c -= lr * self.c_grads[i]
                self.layers[i + 1].d -= lr * self.d_grads[i]

    def save_weights(self, output_dir, name):
        os.makedirs(output_dir, exist_ok=True)
        layer_sizes = [layer.neurons for layer in self.layers]
        np.savez(os.path.join(output_dir, name),
                 layer_sizes=layer_sizes,
                 **{f"w_{i}": w for i, w in enumerate(self.weights)},
                 **{f"b_{i}": b for i, b in enumerate(self.biases)})
        print(f"Weights saved to {output_dir}/{name}.npz")

    def load_weights(self, filepath):
        data = np.load(filepath)
        layer_sizes = data["layer_sizes"]
        self.layers = []
        for size in layer_sizes:
            self.add_layer(int(size))
        self.weights = [data[f"w_{i}"] for i in range(len(self.layers) - 1)]
        self.biases = [data[f"b_{i}"] for i in range(len(self.layers) - 1)]
        print(f"Weights loaded from {filepath}")

    def print_layers(self):
        for layer in self.layers:
            print(f"Layer {layer.idx}: Neurons: {layer.neurons}")


if __name__ == "__main__":

    """ ----------------------------------------- Creating network ----------------------------------------------"""

    network = Network()
    network.add_linear(1)
    network.add_linear(4, activation=mfunc2)
    network.add_linear(1)
    network.print_layers()

    network.init_weights()

    """ ----------------------------------------- features/labels ----------------------------------------------"""

    num_features = 10000
    x = np.linspace(-10,10,num_features)
    y = x**2

    idx = np.random.permutation(len(x))
    x_shuffle = x[idx]
    y_shuffle = y[idx]

    epochs = 100
    """ ----------------------------------------- training ----------------------------------------------"""

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(x_shuffle)):
            x_i = x_shuffle[i].reshape(1, 1)
            y_i = y_shuffle[i]

            network.compute_gradients(x_i, y_i)
            network.update_weights(lr=0.001)

            total_loss += network.loss(x_i, y_i)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(x):.6f}")

    """ ----------------------------------------- save & test ----------------------------------------------"""

    network.save_weights("weights", "my_model")

    x_test = np.array([[0.5]])
    y_pred = network.forward(x_test)
    print(f"\nInput: {x_test.item()}, Predicted: {y_pred.item():.4f}, Actual: {0.5**2}")
