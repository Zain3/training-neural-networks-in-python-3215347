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

"""Notes for a Multi-Layer Perceptron (MLP): 
- For an XOR, if we plot it's truth table, there's no way we can draw one line through it, on a 2D-plane.
- If we use a multi-layer perceptron, we can solve this problem

Moving on to "Training your Network, section 4/5:
- This whole time, we've hard coded the weights of the neural network. What if we could show the network a lot of examples of say, how an XOR behaves (instead of hard-coding them), so that it can LEARN from those examples
- An algorithm to train multilayer perceptrons is the BACK-PROPAGATION algorithm
- The other reason we need to train a neural network is since Linear Separability is hardly a given
- 3 situations in the spectrum of misclassifying and generalizing: 1) underfitting (misclassifies too often), 2) just right (rarely misclassifies, generalizes well), 3) overfitting (bad at generalizing)

The training Process
- a dataset is a collection of samples with features and labels {X,Y}
- Features: the input data
- Labels: the known categories of each sample
- 3 sets: training, validation, testing. The training algorithm is ONLY used on the training set, the other 2 are used for assessment

The Error Function
- An error function measures how bad a classifier is doing- so a small value is good

Gradient Descent
- A training method to minimize the error function of our neural network
- consists of adjusting the weights to find the minimum error
- Think "going downhill" on the error function to the lowest valley
- A potential problem is we get stuck in the Local Minimum

The Delta Rule
- The simplest form of the algorithm we'll implement is the Delta Rule
- is a simple update formula for adjusting the weights in a single perceptron (a single neuron)
- Considers the following: the output error, one input, and a factor known as the LEARNING RATE
= (uppercase)delta_w_ik = lowercase-eta * (y_k - o_k) * x_ik
- The output detla-W_ik (where i is layer number, k is neuron number) will be positive if the label y_k is higher than the output o_k, and negative if its less than the output oi
- This means when we later update wi, it'll later contribute to making the output closer to the provided label
- The learning rate, lowercase-eta, is a unique constant in the ENTIRE neural network- there's one learning rate for all the neurons- name is because higher values result in larger "leaps" for delta-w (change in w), lower values --> smaller leaps
- too small or large learning rate determines if we can find the MINIMUM in our error function
- notice, graphically, it mimics inertia- if it overshoots a minimum (too large of a step), it'll be drawn back into the "valley"

The Backpropagation Algorithm
- Is a general FORM of the delta rule
- has several requirements on the neuron model
- calculates all weight updates throughout the network 
- this is done by propagating the error back through the layers

Steps (to train a multilayer perceptron with 1 sample)
1. feed a sample to the network
2. calculate the mean squared error
3. calculate the error term of each output neuron
    - lowercaseDelta_k = o_k * (1 - o_k) * (y_k - o_k)
    - "o_k * (1-o_k) is the DERIVATIVE of the sigmoid function
    - where lowercaseDelta_k is the error term for the kth neuron in the OUTPUT LAYER
4. Iteratively calculate the error terms in the hidden layers
    - lowercaseDelta_h = o_h * (1-o_h) * sum for the k in the number of outputs of that neuron(w_kh * lowercaseDelta_k)
    -  this is almost the same as for the output layer, but in the hidden layers, we have no idea about the error since we simply dont know what to expect from the intermediate neurons
    - instead, we include that sum, which includes the error term of the neurons connected to a neuron's output (so these neurons are in the next layer), and we just calculated their error terms, lowercaseDelta_k
    - by doing w_kh * lowercaseDelta_k, we're reacting to the error propagated back through the network in the right proportion, by scaling the error terms by the weights. This means that errors with higher weights will take MORE of the blame, and with lower weights will get less of the blame 
5. Apply the delta rule, (uppercase)delta_w_ik = lowercase-eta * (y_k - o_k) * x_ik
6. Adjust the weights
    - to each weight in the layer w_ij, do w_ij = w_ij + (uppercase delta)-w_ij

- The training curve for Backpropagation looks like a backward S (same as that transistor Vout vs. Vin curve)
    
    """


class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the Perceptron class above.
      Attributes:
        layers: A python list with the number of elements per layer.
        bias: The bias term. The same bias is used for ALL neurons.
        eta: The learning rate."""

    def __init__(self, layers, bias=1.0, eta = 0.5):
        """Return a new MLP object with the specified parameters."""
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = [] # The list of lists of neurons (Perceptron objects)- at the end it'll be a numpy array of numpy arrays
        self.values = []  # The list of lists of output values (useful for propagating the results through the network)
        self.d = [] # The list of lists of erro terms (lowercase deltas)

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]

            if i > 0:  # network[0] is the input layer, so it has no neurons, so leave it empty
                for j in range(self.layers[i]):
                    # for every neuron, create a Perceptron with as many inputs as the neurons in the previous layer, not counting the bias value
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))

        # Turn our newly created lists into numpy arrays
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

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
    
    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x,dtype=object) # where x is the feature vector
        y = np.array(y,dtype=object) # y is the label vector

        # Challenge: Write the Backpropagation Algorithm.
        # Here you have it step by step:

        # STEP 1: Feed a sample to the network
        outputs = self.run(x)

        # STEP 2: Calculate the MSE
        error = (y - outputs) # subtraction of vectors
        MSE = sum(error ** 2) / self.layers[-1] # divided by number of neurons in previous layer

        # STEP 3: Calculate the output error term
        self.d[-1] = outputs * (1 - outputs) * (error) # notice how result goes to the last element of our "d" array

        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error
        
        # STEP 5 & 6: Calculate the delta and update the weights
        for i in range(1,len(self.network)): # goes through the layers
            for j in range(self.layers[i]): # goes through the nurons
                for k in range(self.layers[i-1]+1): # goes through the inputs= the "+1" is for the bias
                    if k == self.layers[i-1]: # for the very last weight in a layer, multiply by the bias
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k] # delta = learning rate * error term * input (from previous layer) 
                    self.network[i][j].weights[k] += delta
        return MSE


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

mlp = MultiLayerPerceptron(layers = [2,2,1]) # choose 2 2 1- the first 2 means 2 inputs, second 2 is two perceptrons (one weighted for NAND, one weighted for NOR), and third 1 means one perceptron weighted as an AND
mlp.set_weights([[[-10,-10,15], [15,15,-10]], [[10, 10, -15]]])
#mlp.printWeights()
print("MLP:")
print(f'0 0 = {mlp.run([0,0])[0]:.10f}')
print(f'0 1 = {mlp.run([0,1])[0]:.10f}')
print(f'1 0 = {mlp.run([1,0])[0]:.10f}')
print(f'1 1 = {mlp.run([1,1])[0]:.10f}')


# ------------------------------------------------------------------------------------#

# --------- Test code for Training using Backpropagation and Gradient Descent -------- #
mlp = MultiLayerPerceptron(layers=[2,2,1])
print("\nTraining Neural Network as an XOR gate...\n")
for i in range(3000): # We're running it for 3000 epochs
    mse = 0.0
    mse += mlp.bp([0,0],[0])
    mse += mlp.bp([0,1],[1])
    mse += mlp.bp([1,0],[1])
    mse += mlp.bp([1,1],[0])
    mse = mse / 4
    if (i%100 == 0):
        print(mse)

# mlp.print_weights()

print("MLP:")
print(f'0 0 = {mlp.run([0,0])[0]:.10f}')
print(f'0 1 = {mlp.run([0,1])[0]:.10f}')
print(f'1 0 = {mlp.run([1,0])[0]:.10f}')
print(f'1 1 = {mlp.run([1,1])[0]:.10f}')

# ------------------------------------------------------------------------------------ #

#---------- Test Code for 7 segment Display Recognition --------#
epochs = 3000
mlp = MultiLayerPerceptron(layers=[7,7,10])

# Dataset for the 7 to 10 network
print("Training 7 to 10 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp.bp([1,1,1,1,1,1,0], [1,0,0,0,0,0,0,0,0,0])   # 0 pattern
    mse += mlp.bp([0,1,1,0,0,0,0], [0,1,0,0,0,0,0,0,0,0])   # 1 pattern
    mse += mlp.bp([1,1,0,1,1,0,1], [0,0,1,0,0,0,0,0,0,0])   # 2 pattern
    mse += mlp.bp([1,1,1,1,0,0,1], [0,0,0,1,0,0,0,0,0,0])   # 3 pattern
    mse += mlp.bp([0,1,1,0,0,1,1], [0,0,0,0,1,0,0,0,0,0])   # 4 pattern
    mse += mlp.bp([1,0,1,1,0,1,1], [0,0,0,0,0,1,0,0,0,0])   # 5 pattern
    mse += mlp.bp([1,0,1,1,1,1,1], [0,0,0,0,0,0,1,0,0,0])   # 6 pattern
    mse += mlp.bp([1,1,1,0,0,0,0], [0,0,0,0,0,0,0,1,0,0])   # 7 pattern
    mse += mlp.bp([1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,1,0])   # 8 pattern
    mse += mlp.bp([1,1,1,1,0,0,1], [0,0,0,0,0,0,0,0,0,1])   # 9 pattern
    mse = mse/10.0

print("Done!\n")
pattern = [1.2]
print("The number recognized by the 7 to 10 network is", np.argmax(mlp.run(pattern)))

#-----------------------------------------------------------------#


