import numpy as np
import nnfs 
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from timeit import timeit

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Loss import Loss, Loss_CategoricalCrossEntropy
from Activation_Softmax_Loss_Categorical_CrossEntropy import Activation_Softmax_Loss_Categorical_CrossEntropy
from SGD import Optimizer_SGD

nnfs.init()

# Create dataset
X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3) # first layer has 2 inputs and 3 neurons 

activation1 = Activation_ReLU() # activation function for the first layer

dense2 = Layer_Dense(3, 3) # second layer has 3 inputs and 3 neurons (3 inputs because there are 3 outputs in the previous dense layer)

activation2 = Activation_Softmax() # activation function for the second layer, we want to use softmax here because we are doing a classification problem and softmax gives us a probability distribution for each of our output classes

loss_activation = Activation_Softmax_Loss_Categorical_CrossEntropy()

optimizer = Optimizer_SGD()



dense1.forward_pass(X) #applying our inputs in our neural network layer

activation1.forward_pass(dense1.output) #forward pass through our activation function, input is the output of the previous layer

dense2.forward_pass(activation1.output) #output after the activatino functino on the outputs of the first layer become the input of the second layer

loss = loss_activation.forward_pass(dense2.output, y)

# Let's see output of the first few samples:
print(loss_activation.output[: 5 ])
# Print loss value
print('loss: ', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis = 1 )
if len (y.shape) == 2 :
    y = np.argmax(y, axis = 1 )
accuracy = np.mean(predictions == y)

# Print accuracy
print ( 'acc:' , accuracy)

# Backward pass
loss_activation.backward_pass(loss_activation.output, y)
dense2.backward_pass(loss_activation.dinputs)
activation1.backward_pass(dense2.dinputs)
dense1.backward_pass(activation1.dinputs)

# Print gradients
#print(dense1.dweights)
#print(dense1.dbiases)
#print(dense2.dweights)
#print(dense2.dbiases)

optimizer.update_params(dense1)
optimizer.update_params(dense2)


# Forward pas through the network
# Previous implementation, no backward pass, just forward pass and loss calculation
# Uncomment the code below to see the previous implementation without the combined activation and loss class

'''
# Create dataset
X, y = spiral_data(100, 3)

#plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
#plt.show()

dense1 = Layer_Dense(2, 3) # first layer has 2 inputs and 3 neurons 

activation1 = Activation_ReLU() # activation function for the first layer

dense2 = Layer_Dense(3, 3) # second layer has 3 inputs and 3 neurons (3 inputs because there are 3 outputs in the previous dense layer)

activation2 = Activation_Softmax() # activation function for the second layer, we want to use softmax here because we are doing a classification problem and softmax gives us a probability distribution for each of our output classes

loss_function = Loss_CategoricalCrossEntropy()



dense1.forward_pass(X) #applying our inputs in our neural network layer

activation1.forward_pass(dense1.output) #forward pass through our activation function, input is the output of the previous layer

dense2.forward_pass(activation1.output) #output after the activatino functino on the outputs of the first layer become the input of the second layer

activation2.forward_pass(dense2.output) # after the second layer operations in the neurons we apply our softmax activation function

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y) # calculating the loss of our network

lowest_loss = 9999999 # random high initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range ( 10000 ):
    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn( 2 , 3 )
    dense1.biases += 0.05 * np.random.randn( 1 , 3 )
    dense2.weights += 0.05 * np.random.randn( 3 , 3 )
    dense2.biases += 0.05 * np.random.randn( 1 , 3 )

    # forward pass
    dense1.forward_pass(X)
    activation1.forward_pass(dense1.output)
    dense2.forward_pass(activation1.output)
    activation2.forward_pass(dense2.output)
    
    # output of second layer and returns the loss
    loss = loss_function.calculate(activation2.output, y)
    
    # calculcate the accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    if loss < lowest_loss:
        print(f'New set of weights found, iteration: {iteration}, loss: {loss:.3f}, accuracy: {accuracy:.3f}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else: # Revert the weights and biases
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()        
        
'''