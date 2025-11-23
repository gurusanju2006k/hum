# Program to demonstrate Pandas aggregate functions

import pandas as pd

# Creating a DataFrame
df = pd.DataFrame(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    columns = ["maths", "java", "python"]
)
print(df)
print(df.sum())

print(df.agg(['sum', 'min', 'max', 'count', 'size', 'std']))
# Describe function
print(df.describe())
# Groupby operationA
a = df.groupby('java')
print(a.first())
