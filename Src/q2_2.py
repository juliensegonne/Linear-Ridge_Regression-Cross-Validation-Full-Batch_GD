import numpy as np
from typing import List, Tuple
from q2_1 import *
from q1_1 import rmse


def cross_validation_linear_regression(k_folds: int, hyperparameters: List[float],
                                       X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    """
    Perform k-fold cross-validation to find the best hyperparameter for Ridge Regression.

    Args:
        k_folds (int): Number of folds to use.
        hyperparameters (List[float]): List of floats containing the hyperparameter values to search.
        X (np.ndarray): Numpy array of shape [observations, features].
        y (np.ndarray): Numpy array of shape [observations, 1].

    Returns:
        best_hyperparam (float): Value of the best hyperparameter found.
        best_mean_squared_error (float): Best mean squared error corresponding to the best hyperparameter.
        mean_squared_errors (List[float]): List of mean squared errors for each hyperparameter.
    """

    mean_squared_errors = np.array([])

    if k_folds == 1 :                              #in this case, for each lambda, we simply train the model and test it on arbitrary samples
        train_size = round(4/5*np.size(X,0))       #(train 4/5 and test 1/5) and we select the hyperparamet that offers the smallest rmse
        X_train = X[:train_size,:]
        y_train = y[:train_size]
        X_validation = X[train_size:,:]
        y_validation = y[train_size:]
        for Lambda in hyperparameters :
            w = ridge_regression_optimize(y_train,X_train,Lambda)
            y_hat = np.dot(X_validation,w)
            val_score = rmse(y_validation,y_hat)
            mean_squared_errors = np.append(mean_squared_errors,val_score)


    else :
        fold_size = np.size(X,0)/k_folds
        for Lambda in hyperparameters:
            val_score = np.array([])
            for i in range(k_folds):
                X_validation = X[round(fold_size*i):round(fold_size*(i+1)),:]
                y_validation = y[round(fold_size*i):round(fold_size*(i+1))]
                X_train = np.concatenate((X[:round(fold_size*i),:],X[round(fold_size*(i+1)):,:]),axis=0)
                y_train = np.concatenate((y[:round(fold_size*i)],y[round(fold_size*(i+1)):]),axis=0)
                w = ridge_regression_optimize(y_train,X_train,Lambda)
                y_hat = np.dot(X_validation,w)
                val_score = np.append(val_score,rmse(y_validation,y_hat))
            mean_squared_errors = np.append(mean_squared_errors,np.mean(val_score))
    
    best_mean_squared_error = np.min(mean_squared_errors)
    i_lambda = np.argmin(mean_squared_errors)
    best_hyperparam = hyperparameters[i_lambda]  

    return best_hyperparam, best_mean_squared_error, mean_squared_errors.tolist()
