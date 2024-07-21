print('Hussain Ahmed - FA19-BCS-074')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
# Load the data
data = pd.read_csv('regressiondata_final.csv')
X = data[['x1', 'x2', 'x3', 'x4', 'x5']].values
y = data['y'].values
X = np.hstack([np.ones((X.shape[0], 1)), X])
def linear_regression(X, y):
    X_transpose = X.T
    beta = inv(X_transpose @ X) @ X_transpose @ y
    return beta
beta = linear_regression(X, y)
y_pred = X @ beta
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(data['x1'], y, label='Data')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Scatter Plot of x1 vs y')
plt.show()
plt.scatter(data['x1'], y, label='Data')
plt.plot(data['x1'], X @ beta, color='red', label='Regression Line')
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Regression Line for x1')
plt.legend()
plt.show()
mse = np.mean((y - y_pred)**2)
mse
print(f'Mean Squared Error: {mse}')