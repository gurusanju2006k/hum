# Program to demonstrate time series operation using pandas

import pandas as pd
from datetime import datetime
import numpy as np

# Create a time series with minute frequency for the entire month of September 2025
range_date = pd.date_range(
    start='2025-09-01 00:00:00',
    end='2025-09-30 23:59:00',
    freq='T'   # 'T' stands for minutes
)

# Display the generated time series
print("Generated Time Series (Minute Frequency):")
print(range_date)

# Display the type of the first element in the time series
print("\nType of first timestamp in the series:")
print(type(range_date[0]))
