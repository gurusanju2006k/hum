# Program to demonstrate Array functions using NumPy

import numpy as np

# Creating an array
a = np.array([5, 6, 7, 8, 9, 10])
print("The array is:", a)

# To find the sum of all elements
Sum = np.sum(a)
print("The sum is:", Sum)

# To find average
avg = np.mean(a)
print("The average is:", avg)

# To find minimum
min_val = np.min(a)
print("The minimum value is:", min_val)

# To find maximum
max_val = np.max(a)
print("The maximum value is:", max_val)

# To find minimum index
min_index = np.argmin(a)
print("The minimum index is:", min_index)

# To find maximum index
max_index = np.argmax(a)
print("The maximum index is:", max_index)

# To find standard deviation
std = np.std(a)
print("The standard deviation is:", std)

# To find variance
var = np.var(a)
print("The variance is:", var)
