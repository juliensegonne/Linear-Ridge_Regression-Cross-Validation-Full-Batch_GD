# LinearRidgeRegression-CrossValidation-FullBatchGD

This project was completed as part of the INF8245E (Fall 2024) Machine Learning course. The goal of this assignment is to implement and evaluate linear regression and ridge regression models using both analytical solutions and gradient descent.

The dataset involves predicting employee salaries based on their years of experience and scores on a professional test. The assignment focuses on implementing, validating, and assessing these models while exploring hyperparameter tuning and optimization techniques.
Project Structure

The assignment is organized into multiple Python files, each dedicated to a specific task:
## q1_1.py

- Contains helper functions for implementing linear regression:
- Adds a bias column to the dataset matrix.
- Predicts the output y using the data matrix X and parameters w.
- Computes the optimal parameters w using the closed-form solution.
- Calculates the Root Mean Square Error (RMSE) between predictions and ground truth.

## q1_2.py
- Implements the full linear regression model using the functions defined in q1_1.py.
- Evaluates the model by computing the RMSE on the test set.
- Generates scatter plots comparing predicted and actual salaries as a function of experience and test scores.

## q2_1.py

- Implements the closed-form solution for ridge regression by incorporating L2 regularization into the optimization.

## q2_2.py

- Performs k-fold cross-validation to determine the optimal regularization hyperparameter (λ) for ridge regression.
- Calculates the average RMSE for each hyperparameter value and identifies the best-performing one.

## q2_3.py

- Conducts a hyperparameter search for λ in ridge regression.
- Plots RMSE as a function of λ and provides a theoretical explanation of the results.

## q3_1.py

- Implements gradient calculations for both linear regression and ridge regression.

## q3_2.py

- Implements the gradient descent algorithm to optimize regression models:
- Supports both linear and ridge regression.
- Accepts parameters for regression type, learning rate, regularization term, and the number of epochs.

## q3_3.py

- Applies gradient descent to solve linear and ridge regression problems.
- Plots the training loss at each epoch for performance visualization.

## q3_4.py

- Analyzes the effect of the learning rate hyperparameter on the training process.
- Generates plots showing RMSE and training loss as functions of the learning rate for both linear and ridge regression.


# Acknowledgments

This assignment was completed as part of the INF8245E Machine Learning course. The structure of the files and dataset preparation were provided by the course instructors.
