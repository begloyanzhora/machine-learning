import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from gradient_descent import LinearRegression

x, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(x[:, 0], y, color='b', marker='o', s=30)
plt.show()

model = LinearRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

def mse(y_true, y_predicted):
    return np.mean(y_true - y_predicted)

mse_value = mse(y_test, y_predicted)

print(f'mse_value = {mse_value}')
