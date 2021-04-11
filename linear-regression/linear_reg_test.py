import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt
from gradient_descent import LinearRegression
from sklearn.linear_model import LinearRegression as Lr_skl

x, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model_grd = LinearRegression(n_iters=10000)
model_grd.fit(x_train, y_train)
y_predicted_by_grd = model_grd.predict(x_test)

model_skl = Lr_skl()
model_skl.fit(x_train, y_train)
y_predicted_by_skl = model_skl.predict(x_test)

mse_value_grd = mean_squared_error(y_test, y_predicted_by_grd)
mse_value_skl = mean_squared_error(y_test, y_predicted_by_skl)

print(f'mse value by grd = {mse_value_grd}')
print(f'mse value by skl = {mse_value_skl}')
