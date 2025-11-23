# 9. Implement Multiple Linear Regression

import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data_set = pd.read_csv(r"C:\Users\gurus\OneDrive\Desktop\50_Startups.csv")

# Independent variables X and dependent variable Y
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

# Handling categorical data (State column is index 3)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)

x = ct.fit_transform(x)

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict test set results
y_pred = regressor.predict(x_test)

print("Train Score: ", regressor.score(x_train, y_train))
print("Test Score: ", regressor.score(x_test, y_test))

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, marker='o')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual Profit vs Predicted Profit')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--')
plt.show()
