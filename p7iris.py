# Program to demonstrate functions on iris dataset

import pandas as pd
import matplotlib.pyplot as plt

# Load the iris dataset from CSV file
data = pd.read_csv(r'C:\Users\gurus\OneDrive\Desktop\iris.csv')

# Rename columns for clarity
data.columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']

# Display first 5 rows
print("FIRST 5 ROWS:")
print(data.head())

from pandas.api.types import is_numeric_dtype

# Calculate and show statistics for each numeric column
for col in data.columns:
    if is_numeric_dtype(data[col]):
        print(f"\nColumn: {col}")
        print(f"\tMean = {data[col].mean():.3f}")
        print(f"\tStandard deviation = {data[col].std():.3f}")
        print(f"\tMinimum = {data[col].min():.3f}")
        print(f"\tMaximum = {data[col].max():.3f}")

# Select numeric columns
numeric_columns = data.select_dtypes(include='number').columns

# Covariance matrix
print("\nCOVARIANCE MATRIX:")
print(data[numeric_columns].cov())

# Correlation matrix
print("\nCORRELATION MATRIX:")
print(data[numeric_columns].corr())

# Histogram for each numeric column
data[numeric_columns].hist(figsize=(10, 8))
plt.suptitle("Histograms of Numeric Columns")
plt.show()

# Scatter plot between sepal length and petal length
plt.scatter(data['sepal.length'], data['petal.length'])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.show()
