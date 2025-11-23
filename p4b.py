# Program to check type and basic operations on a pandas time series object

import pandas as pd
from datetime import datetime
import numpy as np

# Generate a range of minute timestamps in September 2025
range_date = pd.date_range(
    start='2025-09-01',
    end='2025-09-30',
    freq='min'
)

# Create a DataFrame with the range_date as the 'date' column
df = pd.DataFrame(range_date, columns=['date'])

# Assign random integer values to the 'date' column for demonstration
df['value'] = np.random.randint(0, 100, size=len(range_date))

# Convert each datetime entry to string and display the first ten for verification
string_data = [str(x) for x in range_date]
print(string_data[1:11])
