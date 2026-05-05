import numpy as np

#Layer Initialization
class Layer_Dense:
    #Dense layer of neurons means that every output from the previous layer is an input to the every neuron in the current layer
    
    # Initializing weights and biases 
    # For now these are random just for the sake of constructing a network of neurons
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01 
        self.biases = np.zeros((1, n_neurons))
        #defning regularization strength
        self.weight_regulaizer_l1 = weight_regularizer_l1
        self.weight_regulaizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    # Forward pass
    # Calculating the outputs based on the inputs, weights and biases
    # Basically how our network makes operations forward and learns
    def forward_pass(self, inputs):
        # Remember input values
        self.inputs = inputs
        
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # Backward pass
    def backward_pass(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0 , keepdims = True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)