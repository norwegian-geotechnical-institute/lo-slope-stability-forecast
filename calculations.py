import numpy as np
import math
from pmdarima import auto_arima
import pandas as pd

# Function to fit solar radiation as a cyclic variable
def cyclic_fit(x, a, b, c, d):
    return a * np.sin(b * (x - c)) + d


# Define functions to calculate saturation vapor pressure, actual vapor pressure, and relative humidity
def saturation_vapor_pressure(temperature):
    return 6.11 * math.exp(7.5 * temperature / (237.7 + temperature))

def actual_vapor_pressure(dew_point):
    return 6.11 * math.exp(7.5 * dew_point / (237.7 + dew_point))

def calculate_relative_humidity(row):
    temperature = row['mean(air_temperature P1D)_value']
    dew_point = row['mean(dew_point_temperature P1D)_value']
    e_s = saturation_vapor_pressure(temperature)
    e = actual_vapor_pressure(dew_point)
    return (e / e_s) * 100

def find_arima_order(time_series):
    # Use pmdarima's auto_arima to find the optimal order
    model = auto_arima(time_series, suppress_warnings=True)
    order = model.get_params()['order']
    return order

def add_albedo_values(df):
    # Define a mapping of month and snow depth to albedo values
    # reference: https://doi.org/10.1080/01431161.2017.1320442 
    albedo_values = {
        (1, True): 0.278,  
        (1, False): 0.335,
        (2, True): 0.293,
        (2, False): 0.378,
        (3, True): 0.233,
        (3, False): 0.416,
        (4, True): 0.213,
        (4, False): 0.375,
        (5, True): 0.152,
        (5, False): 0.254,
        (6, True): 0.138,
        (6, False): 0.144,
        (7, True): 0.138,
        (7, False): 0.135,
        (8, True): 0.134,
        (8, False): 0.115,
        (9, True): 0.127,
        (9, False): 0.147,
        (10, True): 0.147,
        (10, False): 0.224,
        (11, True): 0.168,
        (11, False): 0.273,
        (12, True): 0.21,
        (12, False): 0.368,
    }
    
    def get_snow_depth(row, df):
        # Check if Snow_depth value is NaN
        if pd.isnull(row['Snow_depth']):
            # print("row.name:", row.name)  # Print row name for debugging
            # print("Length of DataFrame:", len(df))
             # Reset index to ensure it's numeric and continuous
            df_reset = df.reset_index(drop=True)
            
            # Find the nearest non-NaN value above and below the current row
            nearest_above = df_reset.loc[:row.name, 'Snow_depth'].dropna().iloc[-1] if row.name != 0 else float('inf')
            nearest_below_series = df_reset.loc[row.name:, 'Snow_depth'].dropna()
            nearest_below = nearest_below_series.iloc[0] if not nearest_below_series.empty else float('inf')

            # Check if nearest_below is infinity (indicating end of the DataFrame)
            if nearest_below == float('inf'):
                # If at the end, reverse the DataFrame and perform the operation
                df_reverse = df_reset[::-1]
                nearest_below_reverse = df_reverse.loc[:len(df_reset) - row.name - 1, 'Snow_depth'].dropna().iloc[-1] if row.name != len(df_reset) - 1 else float('inf')
                nearest_above_reverse = df_reverse.loc[len(df_reset) - row.name - 1:, 'Snow_depth'].dropna().iloc[0] if row.name != 0 else float('inf')
                # Use the nearest value from the reversed DataFrame
                nearest_below = nearest_below_reverse if abs(nearest_below_reverse - (len(df_reset) - row.name - 1)) < abs(nearest_above_reverse - (len(df_reset) - row.name - 1)) else nearest_above_reverse

            # Use the nearest value to determine whether snow is present or not
            snow_depth = nearest_above if abs(nearest_above - row.name) < abs(nearest_below - row.name) else nearest_below
        else:
            snow_depth = row['Snow_depth']
        return snow_depth > 0
    
    # Add albedo column with mapped values based on month and snow depth
    df['albedo'] = df.apply(lambda row: albedo_values.get((row['month'], get_snow_depth(row, df)), None), axis=1)
    
    return df
