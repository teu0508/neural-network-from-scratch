import numpy as np

#Stochastic Gradient Descent (SGD)
class Optimizer_SGD:
    
    # Learning rate can be anything, it will judge how much our values change in the neural network
    # We aim to do small changes to the parameters so we can find the optimal value and not overshoot it.
    # However bigger changes allow us to get to a solution faster, but it can also cause us to miss the optimal solution and diverge
    # TODO: Play around with the values and then implement learning rate decay to get the best of both worlds, start with a higher learning rate and then decrease it as we get closer to the optimal solution
    def __init__(self, learning_rate = 1, decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    # Updates the parameters of the layers based on the gradients calculated during the backward pass over and over again until we get a satisfactory result
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                # If layer doesnt have momentum arrays and biases arrays yet create them initialized to all zeros
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)
            
            # Build the weight updates with momentum and learning rates
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            # Same thing for biases
            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            layer.biases_momentum = biases_updates

        # Updates if there is no momentum        
        else:
            weight_updates = -self.learning_rate * layer.dweights
            biases_updates = -self.learning_rate * layer.dbiases

        # Update final weights and biases variables  
        layer.weights += weight_updates
        layer.biases += biases_updates
        
        
    # quick implementation of a changing learning rate    
    def update_params_test(self, layer, epoch, range):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
        if epoch * 2 > range:
            self.learning_rate = 0.2
            
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def post_update_params(self):
        self.iterations += 1
        
        