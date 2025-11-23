# Program to demonstrate common mathematical operations using numpy arrays

import numpy as np

# Create matrix A
A = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
print("Matrix A is:\n", A)

# Square of each element
print("Square of matrix A is:\n", A ** 2)

# Cube of each element
print("Cube of matrix A is:\n", A ** 3)

# Square root of each element
sqrt_A = np.sqrt(A)
print("Square root of each element of A is:\n", sqrt_A)

# Add 0.5 to each element in Row 0
print("Row 0 after adding 0.5:\n", A[0] + 0.5)

# Create matrix B
B = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
print("Matrix B is:\n", B)

# Sum of A and B
C = A + B
print("Sum of A & B is:\n", C)

# Sum using np.add()
print("Sum of A & B using np.add():\n", np.add(A, B))

# Element-wise multiplication
product = A * B
print("Product of A & B is:\n", product)

# Average, mean, standard deviation, variance
print("Average of elements of A:", np.average(A))
print("Mean of elements of A:", np.mean(A))
print("Standard Deviation of elements of A:", np.std(A))
print("Variance of elements of A:", np.var(A))

# Maximum and minimum
print("Maximum element of A:", np.max(A))
print("Minimum element of A:", np.min(A))
print("Maximum in each row:", np.max(A, axis=1))
print("Minimum in each column:", np.min(A, axis=0))

# Shape, size, type
print("Shape of A:", A.shape)
print("Number of elements in A:", A.size)
print("Class type of A:", type(A))

# Various slicing examples
print("A[0, 1]:", A[0, 1])
print("A[:, 2]:", A[:, 2])
print("A[1:3, 1:2]:", A[1:3, 1:2])
