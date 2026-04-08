import numpy as np

class Activation_Softmax:
    
    def __init__(self):
        pass
    
    # Forward pass
    # Gives us a probability distribution for each of our output classes, the sum of all probabilities will be 1
    # The higher probability class will be the predicted class for our input data
    def forward_pass(self, inputs):
        
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def forward2(self, inputs):
        exp_values = []
        
        #numerator
        for input in inputs:
            exp_values.append(np.exp(input - np.max(input))) #we subtract the max input to avoid overflow and value explostion problems due to the nature of the exponential function (numbers as lowe as 1000 alreaedy overflow)
            
        for value in exp_values:
            value /= np.sum(exp_values, axis=1, keepdims=True) #denominator, we divide the numerator by the sum of all the exponentials to get the probabilities
        
        self.output = exp_values
        
    def backward_pass(self, dvalues):
        
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            
            # Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Add sample wise gradient to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            