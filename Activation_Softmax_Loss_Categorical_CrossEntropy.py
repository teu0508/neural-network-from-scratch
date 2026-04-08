import numpy as np
from Activation_Softmax import Activation_Softmax 
from Loss import Loss, Loss_CategoricalCrossEntropy

class Activation_Softmax_Loss_Categorical_CrossEntropy:
    
    #Initializing the activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
        
    def forward_pass(self, inputs, y_true):
        #Output layer activation function
        self.activation.forward_pass(inputs)
        
        #set the output of the activation function as the output of this combined activation and loss function
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true) #calculate the loss and return it
        
    def backward_pass(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # If labels are one-hot encoded, turn them into discrete values
        # If number of dimensions of y_true (ground truth array) is 2 means its an array of one-hot encoded vectors
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # Copy so can modify
        self.dinputs = dvalues.copy()
        
        #Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient (inputs divided by total samples)
        self.dinputs = self.dinputs / samples