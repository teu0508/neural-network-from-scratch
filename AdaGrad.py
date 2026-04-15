import numpy as np

#Stochastic Gradient Descent (SGD)
class Optimizer_AdaGrad:
    
    def __init__(self, learning_rate = 0.9, decay = 0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    # Updates the parameters of the layers based on the gradients calculated during the backward pass over and over again until we get a satisfactory result
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            # If layer doesnt have momentum arrays and biases arrays yet create them initialized to all zeros
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Build the weight updates with squared gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
            
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def post_update_params(self):
        self.iterations += 1
        
        