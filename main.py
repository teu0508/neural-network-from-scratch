import numpy as np
import nnfs 
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

import Layer_Dense
import Activation_ReLU

nnfs.init()


# Create dataset
X, y = spiral_data(100, 3)

#plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
#plt.show()

dense1 = Layer_Dense(2, 3) # first layer has 2 inputs and 3 neurons 

activation1 = Activation_ReLU() # activation function for the first layer

dense1.forward_pass(X) #applying our inputs in our neural network layer

activation1.forward_pass(dense1.output) #forward pass through our activation function, input is the output of the previous layer

print(activation1.output[:5])


