import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q1_1 import rmse, data_matrix_bias
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression

# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

X_train_biased = data_matrix_bias(X_test)

np.random.seed(42)  # For reproducibility
n_features = X_train.shape[1]
initial_w = np.random.normal(0, 1, size=n_features)
initial_b = 0.0

#learning_rate = 0.0001  # You can change this value to get better results
learning_rate = 0.00000001
num_epochs = 1000
ridge_hyperparameter = 0.1 # You can change this value to get better results


"""
#linear regression
w = np.copy(initial_w)
b = initial_b
loss = np.zeros(num_epochs)
for epoch in range (num_epochs) :
    grad_w, grad_b = compute_gradient_simple(X_train,y_train,w,b)
    b -= learning_rate*grad_b
    for j in range(n_features) :
        w[j] -= learning_rate*grad_w[j]
    w_biased = np.concatenate((b,w),axis=0)
    y_hat = np.dot(X_train_biased,w_biased)
    loss[epoch] = rmse(y_hat,y_train)

print(loss[num_epochs-1])
plt.plot(loss)
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.title("Evolution of the loss function in linear regression")
plt.show()
    
"""

limit = 10**8     #a partir de 10**6 la loss finale commence à etre du meme odg qu'avec un learning rate 10**-2 plus faible, a 10**4 converge moins vite mais arrive a une bonne loss

#linear regression with gradient clipping
w = np.copy(initial_w)
b = initial_b
loss = np.zeros(num_epochs)
for epoch in range (num_epochs) :
    grad_w, grad_b = compute_gradient_simple(X_train,y_train,initial_w,initial_b)
    if abs(grad_b) > limit : grad_b = grad_b/abs(grad_b)*limit
    initial_b -= learning_rate*grad_b
    for j in range(n_features) :
        if abs(grad_w[j]) > limit : grad_w[j] = grad_w[j]/abs(grad_w[j])*limit
        initial_w[j] -= learning_rate*grad_w[j]
    w = np.concatenate((initial_b,initial_w),axis=0)
    y_hat = np.dot(X_train_biased,w)
    loss[epoch] = rmse(y_hat,y_train)

print(loss[num_epochs-1])
plt.plot(loss)
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.title("Evolution of the loss function in linear regression with gradient clipping")
plt.show()
    
    

"""
#ridge regression
w = np.copy(initial_w)
b = initial_b
loss = np.zeros(num_epochs)
for epoch in range (num_epochs) :  
    grad_w, grad_b = compute_gradient_ridge(X_train,y_train,w,b,ridge_hyperparameter)
    b -= learning_rate*grad_b
    for j in range(n_features) :
        w[j] -= learning_rate*grad_w[j]
    w_biased = np.concatenate((b,w),axis=0)
    y_hat = np.dot(X_train_biased,w_biased)
    loss[epoch] = rmse(y_hat,y_train)

print(loss[num_epochs-1])
plt.plot(loss)
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.title("Evolution of the loss function in ridge regression")
plt.show()
"""