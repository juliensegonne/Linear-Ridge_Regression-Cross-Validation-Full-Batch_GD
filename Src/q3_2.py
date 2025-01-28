import numpy as np
from q3_1 import compute_gradient_ridge,compute_gradient_simple



def gradient_descent_regression(X, y, reg_type='simple', hyperparameter=0.0, learning_rate=0.01, num_epochs=100):
    """
    Solves regression tasks using full-batch gradient descent.

    Parameters:
    X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
    y (np.ndarray): Target values of shape (n_samples,).
    reg_type (str): Type of regression ('simple' for simple linear, 'ridge' for ridge regression).
    hyperparameter (float): Regularization parameter, used only for ridge regression.
    learning_rate (float): Learning rate for gradient descent.
    num_epochs (int): Number of epochs for gradient descent.

    Returns:
    w (np.ndarray): Final weights after gradient descent optimization.
    b (float): Final bias after gradient descent optimization.
    """
    n = np.size(X,0)
    m = np.size(X,1)
    w = np.zeros(m)
    b = 0

    for k in range (num_epochs) :
        if reg_type == 'simple' :
            grad_w, grad_b = compute_gradient_simple(X,y,w,b)
            b -= learning_rate*grad_b
            for j in range(m) :
                w[j] -= learning_rate*grad_w[j]
        else :
            grad_w, grad_b = compute_gradient_ridge(X,y,w,b,hyperparameter)
            b -= learning_rate*grad_b
            for j in range(m) :
                w[j] -= learning_rate*grad_w[j]
    return w, b
