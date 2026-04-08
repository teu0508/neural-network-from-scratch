import numpy as np

# Activation function ReLU (Rectified Linear Unit)
# This function introduced non-linearity
# Returns the input if its non-negative and returns 0 if its negative        
class Activation_ReLU:
    
    def __init__(self):
        pass
    
    def forward_pass(self, inputs):
        # Remember input values
        self.inputs = inputs
        
        self.output = np.maximum(0, inputs)
        
    # Just the logic behind the ReLU forward pass without using numpy function 
    def forward2(self, inputs):
        self.output = []
        for input in inputs:
            if input > 0:
                self.output.append(input)
            else:
                self.output.append(0)
    
    # Backward pass
    def backward_pass(self , dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0 ] = 0
