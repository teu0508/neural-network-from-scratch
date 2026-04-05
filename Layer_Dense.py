import numpy as np

#Layer Initialization
class Layer_Dense:
    #Dense layer of neurons means that every output from the previous layer is an input to the every neuron in the current layer
    
    # Initializing weights and biases 
    # For now these are random just for the sake of constructing a network of neurons
    def __init__(self, n_inputs, n_neurons):
        pass 
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01 
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    # Calculating the outputs based on the inputs, weights and biases
    # Basically how our network makes operations forward and learns
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
