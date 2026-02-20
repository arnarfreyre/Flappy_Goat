import numpy as np

""" --------------------------------------------- classes --------------------------------------------------"""

class Layer:
    def __init__(self, neurons, idx):
        self.neurons = neurons
        self.idx = idx


class Network:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []

    def add_layer(self, neurons):
        idx = len(self.layers)
        self.layers.append(Layer(neurons, idx))

    def init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i].neurons, self.layers[i + 1].neurons) * 0.1
            b = np.zeros((1, self.layers[i + 1].neurons))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        a = x
        for i in range(len(self.weights)):
            a = a @ self.weights[i] + self.biases[i]
        return a

    def backprop(self,loss):

        dl_dy_pred = np.sqrt(loss)


        return

    def print_layers(self):
        for layer in self.layers:
            print(f"Layer {layer.idx}: Neurons: {layer.neurons}")


""" ----------------------------------------- Creating network ----------------------------------------------"""

network = Network()
network.add_layer(1)
network.add_layer(4)
network.add_layer(1)
network.print_layers()

network.init_weights()


""" ----------------------------------------- features/labels ----------------------------------------------"""

num_features = 1000
x = np.linspace(-1,1,1000)
y = x**2

""" ----------------------------------------- forward pass ----------------------------------------------"""


x_test = x[0].reshape(1, 1)

y_pred = network.forward(x_test)
print(f"\nInput:  {x_test}")
print(f"Output: {y_pred}")

loss = 0.5*(y_pred-y[0])**2
print(f"loss: {loss}")


