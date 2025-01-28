import numpy as np


def compute_gradient_simple(X, y, w, b):
    """
    Compute the gradients of the loss function with respect to w and b for simple linear regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """

    n = np.size(X,0)
    m = np.size(X,1)
    grad_w = np.zeros(m)
    grad_b = 0
    
    for i in range(n) :
        for j in range(m) :
            grad_b += w[j]*X[i][j]
        grad_b += b-y[i]
    grad_b = 2*grad_b/n
    
    for k in range(m) :
        for i in range(n) :
            for j in range(m) :
                grad_w[k] += w[j]*X[i][j]*X[i][k]
            grad_w[k] += (b-y[i])*X[i][k]
        grad_w[k] = 2*grad_w[k]/n
        
    return grad_w, grad_b


def compute_gradient_ridge(X, y, w, b, lambda_reg):
    """
    Compute the gradients of the loss function with respect to w and b for ridge regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.
        lambda_reg (float): Regularization parameter.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """
    n = np.size(X,0)
    m = np.size(X,1)
    grad_w = np.zeros(m)
    grad_b = 0

    for i in range(n) :
        for j in range(m) :
            grad_b += w[j]*X[i][j]
        grad_b += b-y[i]
    grad_b = 2*grad_b/n + 2*lambda_reg*b
    
    for k in range(m) :
        for i in range(n) :
            for j in range(m) :
                grad_w[k] += w[j]*X[i][j]*X[i][k]
            grad_w[k] += (b-y[i])*X[i][k]
        grad_w[k] = 2*grad_w[k]/n + 2*lambda_reg*w[k]
        
    return grad_w, grad_b
