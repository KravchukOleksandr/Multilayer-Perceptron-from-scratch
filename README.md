# Multilayer-Perceptron-from-scratch
*Implementing perceptron from scratch using only numpy*  
*The perceptrone's code is in the my_perceptron.py file, and the example of usage is in perceptron_test.ipynb*

### What tasks can the implemented perceptron handle?  

1) Any regression task (multi input, multi output, linear and non-linear)
2) Any classification task (including basic computer vision tasks)

### Class parameters

- n_neurons: A list of integers representing the number of neurons in each layer.
- act_func: A list of strings representing the activation functions to be used in each layer. Supported activation functions: 'sigmoid', 'relu', 'leaky_relu', 'softmax', 'heaviside', 'default' (no activation function).
- input_len: An integer representing the length of the input data.

### fit method

The fit method is used to train the multilayer perceptron with the given training data. Returns self.  

Parameters:
 - X: The input data, a 2D numpy array where each row represents a sample and each column represents a feature.
 - y: The target data, a numpy array corresponding to the input data X (it will be reshaped to the shape of output layer).
 - epochs: An integer representing the number of epochs.
 - l_rate: A float representing the learning rate.

### predict method

The predict method is used to make predictions based on the trained multilayer perceptron. Returns predicted data.  

Parameters:  
 - X_test: The input data for which predictions are to be made. It should be a 2D numpy array where X_test.shape[1] == X_train.shape[1].

