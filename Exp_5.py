
!pip install openpyxl

import openpyxl

from google.colab import files
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive

!ls /content/drive/MyDrive/Colab\ Notebooks/Student_dataset.xlsx

import pandas as pd
df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Student_dataset.xlsx')
df.head()
# df.describe()

import numpy as np

# Initialize parameters function
def initialize_parameters(layer_dims):
    np.random.seed(3)
    print("Layer Dimensions:", layer_dims)
    parameters = {}
    L = len(layer_dims)
    print("Total No. of layers in NN:", L)

    for i in range(1, L):
        parameters['W' + str(i)] = np.ones((layer_dims[i-1], layer_dims[i])) * 0.1
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters

# Forward propagation functions
def linear_forward(A_prev, W, b):
    Z = np.dot(W.T, A_prev) + b
    return Z

def relu(Z):
    return np.maximum(0, Z)

# Implementing L-layer forward propagation
def L_layer_forward(X, parameters):
    A = X
    caches = []
    L = len(parameters) // 2  # Number of layers
    for i in range(1, L):  # Iterate for hidden layers
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)  # ReLU activation

        cache = (A_prev, W, b, Z)
        caches.append(cache)

    # Output layer (no activation function applied here, could use sigmoid for binary classification)
    W_out = parameters['W' + str(L)]
    b_out = parameters['b' + str(L)]
    Z_out = linear_forward(A, W_out, b_out)
    AL = Z_out  # or apply sigmoid for classification task
    return AL, caches

# Example execution
layer_dims = [2, 2, 1]  # 2 inputs, 2 hidden neurons, 1 output neuron
parameters = initialize_parameters(layer_dims)
print("parameters :",parameters)

# Simulating a single input data point (for example, 'cgpa' and 'profile_score')
# Make sure df is defined with the correct data if you're testing with a dataset
X = np.array([[7.0], [3.5]])  # Example input, replace with your actual data
y_hat, caches = L_layer_forward(X, parameters)

print("Final output:")
print(y_hat)
