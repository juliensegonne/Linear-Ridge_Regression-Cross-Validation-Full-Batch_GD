import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from q2_2 import cross_validation_linear_regression
from q1_1 import data_matrix_bias


# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

X_train_biased = data_matrix_bias(X_train)

best_hyperp, best_mse, mean_squared_errors = cross_validation_linear_regression(kfolds, hyperparams, X_train_biased, y_train)
print(best_hyperp)

plt.plot(hyperparams, mean_squared_errors)
plt.xscale('log')
plt.xlabel('Hyperparameter')
plt.ylabel('RMSE')
plt.title('RMSE as a function of lambda')
plt.grid(True)
plt.show()



#si l'on souhaite faire la phase de test pour vérifier l'entraînement
"""

w = ridge_regression_optimize(y_train, X_train, best_hyperp)
y_hat = np.dot(X_test,w)


plt.scatter([X_test[i][0] for i in range(len(X_test))], y_hat, color='red', label='Predicted salary')
plt.scatter([X_test[i][0] for i in range(len(X_test))], y_test, color='green', label='Actual salary')
plt.xlabel("Expérience")
plt.ylabel("Salaire")
plt.title("Salary as a function of experience")
plt.legend()
plt.show()

plt.scatter([X_test[i][1] for i in range(len(X_test))], y_hat, color='red', label='Predicted salary')
plt.scatter([X_test[i][1] for i in range(len(X_test))], y_test, color='green', label='Actual salary')
plt.xlabel("Expérience")
plt.ylabel("Salaire")
plt.title("Salary as a function of test score")
plt.legend()
plt.show()
"""