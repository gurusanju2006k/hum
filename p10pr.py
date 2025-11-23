# 10. Implement Polynomial Regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_set = pd.read_csv(r"C:\Users\gurus\OneDrive\Desktop\salary_data.csv")

# x = years of experience, y = salary
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Polynomial Regression (degree = 2)
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Sort x values for smooth curves
x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)

# Graph 1 — Linear Regression
plt.scatter(x, y, color="blue")
plt.plot(x_grid, lin_reg.predict(x_grid), color="red")
plt.title("Salary vs Experience (Linear Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Graph 2 — Polynomial Regression
plt.scatter(x, y, color="green")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.transform(x_grid)), color="black")
plt.title("Salary vs Experience (Polynomial Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
