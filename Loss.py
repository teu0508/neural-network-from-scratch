import numpy as np

class Loss:
    
    def __init__(self):
        pass
    
    # Calculates the data and regularization losses
    def calculate(self, output, y):
        
        sample_losses = self.forward_pass(output, y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
    def regularization_loss(self, layer):
    
        regularization_loss = 0
        
        #L1 weight regularization
        #only change if value is greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
        #same thing for l2 weight regularization
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
        #L1 biases regularization
        #only change if value is greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
        #same thing for L2 biases regularization
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
    
class Loss_CategoricalCrossEntropy(Loss):
    
    def forward_pass(self, y_pred, y_true):
        
        samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward_pass ( self , dvalues , y_true ):
        # Number of samples
        samples = len (dvalues)
        
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len (dvalues[ 0 ])
        
        # If labels are sparse, turn them into one-hot vector
        if len (y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
            
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples