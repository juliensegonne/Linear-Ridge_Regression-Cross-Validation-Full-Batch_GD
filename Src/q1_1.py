import numpy as np


# Part A:
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Returns the design matrix with an all one column appended

    Args:
        X (np.ndarray): Numpy array of shape [observations, num_features]

    Returns:
        np.ndarray: Numpy array of shape [observations, num_features + 1]
    """
    observations = np.shape(X)[0]
    num_features = np.size(X[0])
    X.shape = (observations, num_features) # avoid problems with dimension = (n,)  
    X_bias = np.ones((observations, num_features+1))
    X_bias[:,1:num_features+1] = X
    
    return X_bias


# Part B:
def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes $y = Xw$

    Args:
        X (np.ndarray): Numpy array of shape [observations, features]
        w (np.ndarray): Numpy array of shape [features, 1]

    Returns:
        np.ndarray: Numpy array of shape [observations, 1]
    """
    y = np.dot(X,w)
    return y


# Part C:
def linear_regression_optimize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Optimizes MSE fit of $y = Xw$

    Args:
        y (np.ndarray): Numpy array of shape [observations, 1]
        X (np.ndarray): Numpy array of shape [observations, features]

    Returns:
        Numpy array of shape [features, 1]
    """

    X_transpose = X.T
    X_inv = np.linalg.inv(np.dot(X_transpose,X))
    X_inv_transp = np.dot(X_inv,X_transpose)
    w = np.dot(X_inv_transp,y)
    return w


# Part D
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Evaluate the RMSE between actual and predicted values.

    Parameters:
    y (list or np.array): The actual values.
    y_hat (list or np.array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    y = np.array(y)               #convert a list into an array to use "-" after, the arrays stay the same here 
    y_hat = np.array(y_hat)
    n = np.size(y)
    rmse_err = np.sqrt(np.sum((y-y_hat)**2/n))
    
    return rmse_err

