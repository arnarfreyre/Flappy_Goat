import numpy as np
import matplotlib.pyplot as plt
from custom_network import Network

""" ----------------------------------------- load pretrained model ----------------------------------------"""

network = Network()
network.load_weights("weights/my_model.npz")
network.print_layers()

""" ----------------------------------------- test --------------------------------------------------------"""

range_start = -10
range_end = 10
test_inputs = np.linspace(range_start, range_end, 500)
actuals = test_inputs ** 2

preds = []
for val in test_inputs:
    x = np.array([[val]])
    preds.append(network.forward(x).item())
preds = np.array(preds)

""" ----------------------------------------- plot --------------------------------------------------------"""

plt.figure(figsize=(10, 6))
plt.plot(test_inputs, actuals, label="Actual (x²)")
plt.plot(test_inputs, preds, label="Predicted", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.show()
