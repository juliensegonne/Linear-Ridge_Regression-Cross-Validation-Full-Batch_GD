import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from q1_1 import data_matrix_bias, linear_regression_predict, linear_regression_optimize, rmse


# Loading the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

# Write your code here:

# Find the optimal parameters only using the training set

X_train_biased = data_matrix_bias(X_train)
w = linear_regression_optimize(y_train,X_train_biased)

# Report the RMSE and Plot the data on the test set
X_test_biased = data_matrix_bias(X_test)
y_hat = linear_regression_predict(X_test_biased,w)
rmse_err = rmse(y_test,y_hat)
print(rmse_err)


plt.scatter([X_test[i][0] for i in range(len(X_test))], y_hat, color='red', label='Predicted salary')
plt.scatter([X_test[i][0] for i in range(len(X_test))], y_test, color='green', label='Actual salary')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary as a function of experience")
plt.legend()
plt.show()

plt.scatter([X_test[i][1] for i in range(len(X_test))], y_hat, color='red', label='Predicted salary')
plt.scatter([X_test[i][1] for i in range(len(X_test))], y_test, color='green', label='Actual salary')
plt.xlabel("Test score")
plt.ylabel("Salaire")
plt.title("Salary as a function of test score")
plt.legend()
plt.show()