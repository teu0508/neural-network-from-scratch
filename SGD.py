

#Stochastic Gradient Descent (SGD)
class Optimizer_SGD:
    
    # Learning rate can be anything, it will judge how much our values change in the neural network
    # We aim to do small changes to the parameters so we can find the optimal value and not overshoot it.
    # However bigger changes allow us to get to a solution faster, but it can also cause us to miss the optimal solution and diverge
    # TODO: Play around with the values and then implement learning rate decay to get the best of both worlds, start with a higher learning rate and then decrease it as we get closer to the optimal solution
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate
        
    # Updates the parameters of the layers based on the gradients calculated during the backward pass over and over again until we get a satisfactory result
    def update_params(self, layer):
        layer.weights += - self.learning_rate * layer.dweights
        layer.biases += - self.learning_rate * layer.dbiases
        