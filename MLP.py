import numpy as np

"""Notes
- the reasons Perceptrons work as Classifiers is because we use data sets that are Linearly Separable
- data of 2 dimensions (2 data points)- separated by a line. Data of 3 points- by a plane. Data of more than 4 points- by a HYPER-PLANE
- its a common pratice to tie the bias INPUTS to 1, but we can change / only deal with its WEIGHT

"""


class Perceptron:
    """A single neuron with the sigmoid activation function.
    Attributes:
      inputs: The number of inputs in the perceptron, not counting the bias.
      bias: The bias term. By default, it's 1.0."""

    def __init__(self, inputs, bias=1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias term)"""
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input Values."""
        # insert bias at end of input array since we'll use it as the last input
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        # plug sum into activation function
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights, w_init is a python list with the weights"""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))


# ------------------------------------------------------------------#

"""For an XOR, if we plot it's truth table, there's no way we can draw one line through it, on a 2D-plane.
If we use a multi-layer perceptron, we can solve this problem"""


class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the Perceptron class above.
      Attributes:
        layers: A python list with the number of elements per layer.
        bias: The bias term. The same bias is used for ALL neurons.
        eta: The learning rate."""

    def __init__(self, layers, bias=1.0):
        """Return a new MLP object with the specified parameters."""
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.network = [] # The list of lists of neurons (Perceptron objects)- at the end it'll be a numpy array of numpy arrays
        self.values = []  # The list of lists of output values (useful for propagating the results through the network)

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]

            if i > 0:  # network[0] is the input layer, so it has no neurons, so leave it empty
                for j in range(self.layers[i]):
                    # for every neuron, create a Perceptron with as many inputs as the neurons in the previous layer, not counting the bias value
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))

        # Turn our newly created lists into numpy arrays
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.network], dtype=object)

    def set_weights(self, w_init):
        """Set the weights.
           w_init is a list of lists with the weights for all but the input layer."""
        for i in range(len(w_init)):  # iterate through the layers in the network
            for j in range(len(w_init[i])): # iterate through the neurons in each layer
                # w_init will have one list entry in the 1st dimension because we're not specifying for the input layer, as it has no neurons (so i+1)
                self.network[i+1][j].set_weights(w_init[i][j])

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron"""
        x = np.array(x, dtype=object)
        self.values[0] = x
        # iterate through the neural network, feeding each layer the values from the previous layer
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]
# ------------------------------------------------------------------------#


# ------------ test code to test a SINGLE Perceptron as an AND gate --------------#
neuron = Perceptron(inputs=2)
neuron.set_weights([10, 10, -15])  # AND

# our threshold for the neuron "firing" is when the neural network outputs greater than 0.5. input (1 1) is the only one that gives above 0.5, so it works
print("Gate:")
print(f'0 0 = {neuron.run([0,0]):.10f}')
print(f'0 1 = {neuron.run([0,1]):.10f}')
print(f'1 0 = {neuron.run([1,0]):.10f}')
print(f'1 1 = {neuron.run([1,1]):.10f}')


# --------------- Test code to test MULTI-LAYER Perceptron as an XOR gate ----------------------- #

mlp = MultiLayerPerceptron(layers = [2,2,1]) # choose 2 2 1 as since its 2 for NAND, 2 for OR, 1 for AND
mlp.set_weights([[[-10,-10,-15], [15,15,-10]], [[10, 10, -15]]])
#mlp.printWeights()
print("MLP:")
print(f'0 0 = {mlp.run([0,0])[0]:.10f}')
print(f'0 1 = {mlp.run([0,1])[0]:.10f}')
print(f'1 0 = {mlp.run([1,0])[0]:.10f}')
print(f'1 1 = {mlp.run([1,1])[0]:.10f}')


# ------------------------------------------------------------------------------------#