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


# Hyperparameters
num_epochs = 1000
ridge_hyperparameter = 0.1
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Different learning rates to try

np.random.seed(42)  # For reproducibility
n_features = X_train.shape[1]
initial_w = np.random.normal(0, 1, size=n_features)
initial_b = 0.0

last_loss = np.array([])
limit = 10**3
"""
#linear regression
for learning_rate in learning_rates :
    w = np.copy(initial_w)
    b = initial_b
    loss = np.zeros(num_epochs)
    for epoch in range (num_epochs) :
        grad_w, grad_b = compute_gradient_simple(X_train,y_train,w,b)
        if abs(grad_b) > limit : grad_b = grad_b/abs(grad_b)*limit
        b -= learning_rate*grad_b
        for j in range(n_features) :
            if abs(grad_w[j]) > limit : grad_w[j] = grad_w[j]/abs(grad_w[j])*limit
            w[j] -= learning_rate*grad_w[j]
        w_biased = np.concatenate((b,w),axis=0)
        y_hat = np.dot(X_train_biased,w_biased)
        loss[epoch] = rmse(y_hat,y_train)
    last_loss = np.append(last_loss,loss[num_epochs-1])

    plt.plot(loss)
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.title("Evolution of the loss function in linear regression")
    plt.show()

plt.plot(learning_rates,last_loss,marker='o')
plt.xscale('log')
plt.ylabel("RMSE")
plt.xlabel("Learning rate")
plt.title("RMSE versus Learning rate (linear regression)")
plt.show()

"""
#ridge regression

for learning_rate in learning_rates :
    w = np.copy(initial_w)
    b = initial_b
    loss = np.zeros(num_epochs)
    for epoch in range (num_epochs) :  
        grad_w, grad_b = compute_gradient_ridge(X_train,y_train,w,b,ridge_hyperparameter)
        if abs(grad_b) > limit : grad_b = grad_b/abs(grad_b)*limit
        b -= learning_rate*grad_b
        for j in range(n_features) :
            if abs(grad_w[j]) > limit : grad_w[j] = grad_w[j]/abs(grad_w[j])*limit
            w[j] -= learning_rate*grad_w[j]
        w_biased = np.concatenate((b,w),axis=0)
        y_hat = np.dot(X_train_biased,w_biased)
        loss[epoch] = rmse(y_hat,y_train)
    last_loss = np.append(last_loss,loss[num_epochs-1])

    plt.plot(loss)
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.title("Evolution of the loss function in ridge regression")
    plt.show()


plt.plot(learning_rates,last_loss,marker='o')
plt.xscale('log')
plt.ylabel("RMSE")
plt.xlabel("Learning rate")
plt.title("RMSE versus Learning rate (ridge regression)")
plt.show()