import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
import matplotlib.pyplot as plt
import urllib
import urllib.request
import os
from zipfile import ZipFile
import cv2 #Real code now
import time
import copy
import pickle

#nnfs.init()

'''
Dumb Ideas:

loss = 1-accuracy

When loss reaches an asymptote, multiply the loss by a scale factor and see what happens


Notes:

Learning rate decay = make learning rate decay exponentially to stay in a minimum

Momentum = save gradient after each batch and descend average gradient
using momentum + learning rate: step vector = -1 * batch_vector * learning_rate  + previous step vector * momentum
(previous step will be -1 * gradient vector so no need for double negative)


-is it better or worse than
gradient descent using gradient vector of entire training set?

AdaGrad - "adaptive gradient"
"AdaGrad provides a way to normalize parameter updates by keeping a history of previous updates — the bigger the
sum of the updates is, in either direction (positive or negative), the smaller updates are made further in training.
This lets less-frequently updated parameters to keep-up with changes, effectively utilizing more neurons for training."

RMSProp - "Root Mean Square Propagation":
same idea as AdaGrad but calculated using a more effective formula

L1 + L2 regularization:

Goal: to penalize the network when it learns to use weights and biases that are "too strong" or of a very high relative
magnitude to other paramaters of the network. Works by adding penalty values (L1 and/or L2) to the loss/cost which are higher
when the network's parameters are of higher magnitude

L1:
    -sum of abs of all weights * a constant 
    -sum of abs of all biases * a constant 
L2:
    -sum of squares of all weights * a constant
    -sum of squares of all biases * a constant

Dropout:

Goal: to prevent a neural network from becoming too dependent on any neuron or
for any neuron to be relied upon entirely in a specific instance. Also prevents memorizing
of the training data as well as co-adoption, which happens when neurons depend on the
output values of other neurons and do not learn the underlying function on their own.

A coefficent determins the proportion of neurons which are randomly disabled each iteration.
The program will use a numpy binomial distrubtion function to determine which neurons will
be multiplied by 0 to zero their values, effectively turning them off for the forward pass.

Problem: If only half the neurons are used, the values of the neurons in each layer after the
first, which sum the previous activations * weights + bias, will effectively halved, changing
the behavior of the network. Dropout should obviously not be used in predicting, so in order
to keep the magnitude of the neuron values on a consistent scale, the neuron values should be
scaled back up proportionally to the dropout coefficent.
ex.) dropout_rate = 0.5, output for one kept neuron = activated_neuron / (1-dropout_rate)

Binary Logistic Regression:
an alternate output layer option, where each neuron separately represents two classes — 0 for
one of the classes, and a 1 for the other.

uses sigmoid (which I already know) as activation function vs softmax and binary cross entropy
to calculate loss

great explanation:
https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

IMPORTANT NOTE ABOUT ADAM:
learning_rate=0.005, decay=5e-5 seem to be better on the spiral dataset, it seems the book suggests
a default learning rate which is WAY too low. (0.001)

When switching to those parameters, the binary logistic regression model performed way better than even
in the book. (validation, acc: 0.980, loss: 0.112 vs. validation, acc: 0.945, loss: 0.207)

optimizer values: 
#optimizer = Optimizer_SGD(decay=1e-3,momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-6, rho=0.999)

Regression vs. Classification:
Since I'm moving on to regression in the book I thought I would write this down so people do not think I'm
dumb: yes I know the difference between classification, which classifies inputs (outputs a classification),
and regression, which predicts a scalar output value with given inputs.

Note about accuracy_precision:

This value dictates how close an output can be to the correct output before it is considered to be correct,
or usable. The accuracy is calculated as the average of 1s and 0s for each output neuron where a 1 means
that the predicted value was close enough to be considered correct.

Notes about training classifier in batches:

make sure each minibatch has even (or at least random) amount of cases of each class
according to the book "batch sizes range between 32 and 128 samples"

Calculate loss + accuracy on the single minibatch as well as the full batch (full epoch)

Note on syntax: * seperates args from kwargs

META NOTES:
The magnitude of a training step must be proportional to loss. Anything else would theoretically be a
step away from the global minimum. Why is learning rate decay and parameter regularization anything more than a method of
compensation for a bad loss function?

Why should the total loss of a batch be divided by the count of samples in a batch?

Humans seem to learn to memorize data the way a neural net is trained to fit data in order to mathematically represent complex
patterns. People do not seem to learn patterns in this way. Rather, they seem to have the capacity to understand patterns in a
much faster, more calculatable way, at a subconcious level. Wait this is so confusing because there's concious vs subconcious,
memorizing versus pattern recognition. This is a much deeper problem which I must explore. The result could be a better loss
function. 
'''
# Makes coding easier to consider inputs as a layer with a forward method 
# which just outputs the input values
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                         bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Todo: switch to np.random.uniform in range [-0.1, 0.1]
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout:
    def __init__(self, rate):
        #invert rate from prop. of neurons to drop to neurons to keep
        self.rate = 1 - rate
        
    def forward(self, inputs, training):
        self.inputs = inputs
        #If not training, don't dropout
        if not training:
            self.output = inputs.copy()
            return
        # Save input values
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
        
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs
        
class Activation_ReLU:
    
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Copy since original variable going to get modified
        self.dinputs = dvalues.copy()
        # Zero gradient where inpuut values were negative
        self.dinputs[self.inputs <= 0] = 0
        # Don't use please
    def predictions(self, outputs):
        return outputs
    
class Activation_Softmax:
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten
            single_output = single_output.reshape(-1, 1)
            # Jacobian
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # if one-hot encoded, turn into discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        
class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Linear:
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
              self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
              
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
          
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
      # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    # If layer does not contain cache arrays,
     # create them filled with zeros
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected /  (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
        
class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):
        # determine reg. loss to add as explained in notes
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 *np.sum(layer.biases * layer.biases)

        return regularization_loss
    
        
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        # (mean)
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
            
        return data_loss, self.regularization_loss()
        
    # For resetting accumulated loss after an epoch
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip values to avoid error with ln(0) returning infinity
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # if each element in y_true corresponds to the class index of the correct guess in
        # the output. (not one-hot encoded)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # elif one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # num samples
        samples = len(dvalues)
        # num labels/sample, use 1st sample to count
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calc gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss): # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
         
         # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
    
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                           (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true-y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true-y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.abs(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
  

# All designed to  be run in batches of inputs, to compute single output, make batch with only one
# set.
class Model: 
    def __init__(self):
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        # * signifies keyword args
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            # If first layer, previous layer object = input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
                
        if self.loss is not None:
              self.loss.remember_trainable_layers(self.trainable_layers)
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # After iteration layer is last layer
        return layer.output

    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)  
        
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        #Making epochs start on 1, lol
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    # example: epoch 0 step 0: batch_X = X from 0 to 128
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                
                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)
                
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                            f'acc: {accuracy:.3f}, ' +
                            f'loss: {loss:.3f} (' +
                            f'data_loss: {data_loss:.3f}, ' +
                            f'reg_loss: {regularization_loss:.3f}), ' +
                            f'lr: {self.optimizer.current_learning_rate}')
                    
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' +
                    f'acc: {epoch_accuracy:.3f}, ' +
                    f'loss: {epoch_loss:.3f} (' +
                    f'data_loss: {epoch_data_loss:.3f}, '+
                    f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')
            ##--{Validate After Each Epoch}--##
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)
            
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()
        
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            output = self.forward(batch_X, training=False) 
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
            
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')
        
    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
                
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
              # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to the list of predictions
            output.append(batch_output)
        return np.vstack(output)
        
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
        
    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(x):
                prediction_steps += 1
        output = []
        for i in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)
                
#--{End Library}--#
'''
URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}...")
    urllib.request.urlretrieve(URL, FILE)

print("Unzipping images...")
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print("Done.")
'''
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):
      # Load both sets separately
      X, y = load_mnist_dataset('train', path)
      X_test, y_test = load_mnist_dataset('test', path)
      # And return all the data
      return X, y, X_test, y_test

# ONLY RUN IF NOT IMPORTED!!!!
def main():
    print("Running numpynet.py main function")
    
    X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)

    X = X[keys]
    y = y[keys]

    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    fashion_mnist_labels = {
          0: 'T-shirt/top',
          1: 'Trouser',
          2: 'Pullover',
          3: 'Dress',
          4: 'Coat',
          5: 'Sandal',
          6: 'Shirt',
          7: 'Sneaker',
          8: 'Bag',
          9: 'Ankle boot'
      }
    
    #model = Model.load("test.model")
    model = Model()
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-8), accuracy=Accuracy_Categorical())
    model.finalize()
    model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
    model.save("giga_train.model")
    '''
    image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(image_data, cmap='gray')
    plt.show()
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    plt.imshow(image_data, cmap='gray')
    plt.show()
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    confidences = model.predict(image_data)
    predictions = model.output_layer_activation.predictions(confidences)

    for prediction in predictions:
        print(fashion_mnist_labels[prediction])
    '''


    
if __name__ == "__main__":
    main()
