import numpy as np
def sigmoid_function(x):
  return 1/(1+np.exp(-x))

def derivative(x):
  return x*(1-x)

x1 = 0.4
x2 = 0.8

y_actual = 1

w1,w2,w3,w4 = np.random.rand(4)
b1,b2 = np.random.rand(2)

learning_rate = 0.1
epochs = 100
losses = []
iterations = 0

for iterations in range(epochs):
  #forward propagation
  # for hidden layer
  z_hid = w1*x1 + w2*x2 + b1
  h = sigmoid_function(z_hid)

  #for output layer
  z_out = w3*h + w4*h + b2
  y_pred = sigmoid_function(z_out)

  #error calculation
  E = (1/2) * (y_actual - y_pred) * 2
  losses.append(E)

  #Backpropagation
  dE_dypred = -(y_actual - y_pred)
  dypred_dh = derivative(y_pred) * (w3 + w4)
  dh_dw1 = derivative(h) * x1
  dh_dw2 = derivative(h) * x2

  #gradients calculation
  dw3 = dE_dypred * derivative(y_pred) * h
  dw4 = dE_dypred * derivative(y_pred) * h
  dw1 = dE_dypred * dypred_dh * dh_dw1
  dw2 = dE_dypred * dypred_dh * dh_dw2
  db1 = dE_dypred * dypred_dh * derivative(h)
  db2 = dE_dypred * derivative(y_pred)

  #update weight and bias
  w1 = w1 - learning_rate * dw1
  w2 = w2 - learning_rate * dw2
  w3 = w3 - learning_rate * dw3
  w4 = w4 - learning_rate * dw4
  d1 = b1 - learning_rate * dw1
  d2 = b2 - learning_rate * dw2

  

  if iterations % 100 == iterations:
    iterations = iterations + 1
    print(f"Epochs: {iterations}, Loss: {E}")
  iterations += 1
