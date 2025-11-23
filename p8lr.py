# 8. Implement simple linear regression.

import pandas as pd

dataset = pd.read_csv(r"C:\Users\gurus\OneDrive\Desktop\salary_data.csv")

# x = Years of experience (first column)
# y = Salary (last column)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

from sklearn.model_selection import train_test_split

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)
x_pred = regressor.predict(X_train)

import matplotlib.pyplot as plt

# Training Set Plot
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, x_pred, color='red')
plt.title('Salary vs Experience (Train Dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (In Rupees)')
plt.show()

# Test Set Plot
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, x_pred, color='red')  # same line
plt.title('Salary vs Experience (Test Dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (In Rupees)')
plt.show()
