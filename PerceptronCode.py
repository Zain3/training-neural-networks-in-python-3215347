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
    
    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias term)"""
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias
    
    def run(self, x):
        """Run the perceptron. x is a python list with the input Values."""
        # insert bias at end of input array since we'll use it as the last input
        x_sum = np.dot(np.append(x,self.bias), self.weights)
        # plug sum into activation function
        return self.sigmoid(x_sum)
    
    def set_weights(self, w_init):
        """Set the weights, w_init is a python list with the weights"""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))
    

# test code to test Perceptron as an AND gate
neuron = Perceptron(inputs=2)
neuron.set_weights([10, 10, -15]) # AND

# our threshold for the neuron "firing" is when the neural network outputs greater than 0.5. input (1 1) is the only one that gives above 0.5, so it works
print("Gate:")
print(f'0 0 = {neuron.run([0,0]):.10f}')
print(f'0 1 = {neuron.run([0,1]):.10f}')
print(f'1 0 = {neuron.run([1,0]):.10f}')
print(f'1 1 = {neuron.run([1,1]):.10f}') 
