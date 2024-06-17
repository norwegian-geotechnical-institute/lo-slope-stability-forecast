import os
import joblib
from sklearn.metrics import mean_squared_error,  r2_score
import pandas as pd
import pastas as ps
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.keyvault.secrets import SecretClient


class BlobClientLoader:
    def __init__(self, account_url: str, container_name: str):
        default_credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        self._container_client = blob_service_client.get_container_client(container_name)

    def load_blob(self, src_blob: str, dst_file: str):
        with open(file=dst_file, mode="wb") as download_file:
            download_file.write(self._container_client.download_blob(src_blob).readall())
class KeyvaultSecretClient:
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=vault_url, credential=credential)

    def get_secret(self, name: str):
        return self._client.get_secret(name).value


class EnvSecretClient:
    def get_secret(self, name: str):
        return os.environ[name]


def load_model(filename: str, blob_client_loader: BlobClientLoader | None):
    if blob_client_loader:
        models_dir = "tmp_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        blob_client_loader.load_blob(filename, os.path.join(models_dir, filename))
        return joblib.load(os.path.join(models_dir, filename))
    else:
        return joblib.load(filename)
    
def VWC_RF_predictions(merged_df,loaded_model):
    #print(merged_df)
   
    # Define input and output variables
    X_forecast = merged_df[[ 'Precipitation','AirTemperature','Wind_speed', 'LAI', 'solar_radiation','RelativeHumidity','Snow_depth','albedo']]
    # Interpolate NaN values using linear interpolation
    X_forecast = X_forecast.interpolate(method='linear', limit_direction='both')

    # fill remaining NaN values with the last available value:
    X_forecast.fillna(method='ffill', inplace=True)
    # Rename columns in X_forecast
    X_forecast = X_forecast.rename(columns={
        'Precipitation':'sum(precipitation_amount P1D)_value',
        'AirTemperature': 'mean(air_temperature P1D)_value',
        'Wind_speed': 'mean(wind_speed P1D)_value',
        'RelativeHumidity': 'relative_humidity',
        'Snow_depth': 'snow_depth'
         })
    #print(X_forecast)

    # Make forecasts for Vol moist content and pore pressure
    y_forecasts = loaded_model.predict(X_forecast)

        #print(y_forecasts_original_scale)
    return y_forecasts
# In[4]:

def Pastas_predictions(merged_df,forecast_df_mean_per_day,specified_duration):
    #print(merged_df.columns)

    merged_df['date'] = pd.to_datetime(merged_df['mid_time'], utc=True).dt.date
    #print(merged_df)
    # Check for duplicate values in the 'date' column
    duplicate_mask = merged_df.duplicated(subset='date', keep='last')

    # Keep the last row for each duplicate date and delete other rows
    merged_df = merged_df[~duplicate_mask].sort_values(by='mid_time').reset_index(drop=True)
    
    # Check for discontinuity in dates
    date_range = pd.date_range(start=merged_df['date'].min(), end=merged_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(merged_df['date'])]
    #print(missing_dates)
    
    # Add rows corresponding to missing dates and interpolate values
    for missing_date in missing_dates:
        if missing_date.date() in merged_df['date'].values:
            continue  # Skip existing dates
        else:
            #print(missing_date.date())  # Print the date without the timestamp
            missing_date = missing_date.date()

            # Find the closest dates before and after the missing date with a timedelta of 1 day
            date_before = (pd.to_datetime(missing_date) - pd.Timedelta(days=1)).date()
            date_after = (pd.to_datetime(missing_date) + pd.Timedelta(days=1)).date()
            #print(date_before,date_after)

            
            # Find the index of 'before' or increment the timedelta until it's found
            index_before = None
            delta = 1
            while index_before is None:
                try:
                    date_before_candidate = (pd.to_datetime(missing_date) - pd.Timedelta(days=delta)).date()
                    index_before = merged_df[merged_df['date'] == date_before_candidate].index[0]
                except IndexError:
                    delta += 1
            #print(delta)

            # Find the index of 'date_after' 
            index_after = index_before+1
            #print(index_before,index_after)
            # Interpolate values for each column using linear interpolation
            interpolated_values = {}
            for col in merged_df.columns:
                if col == 'mid_time':
                    interpolated_values[col] = pd.to_datetime(f"{missing_date} 00:00:00", utc=True)
                elif col == 'date':
                    interpolated_values[col] = missing_date
                else:
                    x = [index_before, index_after]
                    #print(x)
                    # Select rows by their integer position using .iloc[]
                    y = [merged_df.iloc[index_before][col], merged_df.iloc[index_after][col]]
                    #print(y)
                    interpolated_values[col] = np.interp((index_before + ((delta) / (index_after - index_before))), x, y)

            # Add a new row with the current index of 'date_after'
            merged_df = pd.concat([merged_df.loc[:index_after], pd.DataFrame([interpolated_values], index=[index_after]), merged_df.loc[index_after + 1:]]).sort_index()



    # Convert 'mid_time' to datetime in UTC
    merged_df['mid_time'] = pd.to_datetime(merged_df['mid_time'], utc=True)

    # Sort the DataFrame by 'mid_time'
    merged_df = merged_df.sort_values(by='mid_time').reset_index(drop=True)
    #print(merged_df.columns)
    #print(merged_df)
    days=(specified_duration/24)+1
    # Define the column names and corresponding series names
    columns_to_series = {
        'DL1_WC1': 'vwcdata1',
        'DL1_WC2': 'vwcdata2',
        'DL1_WC3': 'vwcdata3',
        'DL1_WC4': 'vwcdata4',
        'DL1_WC5': 'vwcdata5',
        'DL1_WC6': 'vwcdata6',
        'PZ01_D': 'ppdata1',
        'AirTemperature': 'temp',
        'Precipitation': 'precip',
        'RelativeHumidity': 'hum',
        'solar_radiation': 'sol',
        'LAI': 'lai',
        'Wind_speed':'ws',
        'Snow_depth':'sd',
        'albedo':'albedo'
    }
    
    # Create Series from columns in merged_df
    for column, series_name in columns_to_series.items():
        series = merged_df[['mid_time', column]].copy()  # Select both columns
        #print(series['mid_time'])
        series['mid_time'] = pd.to_datetime(series['mid_time'], utc=True).dt.date  # Extract only the date (in UTC)
        series.set_index('mid_time', inplace=True)  # Set 'mid_time' as the index
        series.index = pd.to_datetime(series.index)  # Ensure datetime format without time zone (pastas will fail if time zone aware)
        series.index.freq='D'
        globals()[series_name] = series[column]  
    #print(vwcdata1.index)

    # # #print(series)
    # import matplotlib.pyplot as plt
    # # Combine all series into a single DataFrame
    # df = pd.DataFrame({'P (mm)': precip, 'T (°C)': temp, 'SoR (kW/m²)': sol, 'LAI': lai, 'RH (%)': hum, 'WS (m/s)': ws, 'SD (m)': sd, 'Albedo': albedo})

    # # Set Arial Bold as the font globally
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.weight'] = 'bold'

    # # Increase the length of y-axes by adjusting figsize and gridspec_kw
    # fig, axes = plt.subplots(nrows=len(df.columns), sharex=True, figsize=(10, 14), gridspec_kw={'height_ratios': [1.5] * len(df.columns)})

    # for i, col in enumerate(df.columns):
    #     axes[i].plot(df.index, df[col], label=col, color='darkblue')
    #     axes[i].set_ylabel(col, fontsize=14,fontweight='bold')  # Set font size for the y-axis label
    #     axes[i].tick_params(axis='both', labelsize=14)  # Set font size for tick labels

    # # Customize the plot
    # axes[-1].set_xlabel('Date', fontsize=14,fontweight='bold')  # Set font size for the x-axis label

    # # Remove the gap between subplots
    # plt.subplots_adjust(hspace=0)

    # # Save the plot with 600 dpi
    # plt.savefig('pastas_input_1.png', dpi=600)

    # # Show the plot
    # plt.show()

    # Define the column names and corresponding series names
    columns_to_series = {
        'VWC_0.1m': 'vwcdata1',
        'VWC_0.5m': 'vwcdata2',
        'VWC_1.0m': 'vwcdata3',
        'VWC_2.0m': 'vwcdata4',
        'VWC_4.0m': 'vwcdata5',
        'VWC_6.0m': 'vwcdata6',
        'PP_6.0m': 'ppdata1'
    }
    max_trials=5
    # Iterate over the series names
    for series_name, column in columns_to_series.items():
        trials = 0
        r2 = 0
        while trials < max_trials and r2 < 0.8:
            # Create a model object by passing it the observed series
            ml = ps.Model(globals()[column], name=series_name)

            # Add the rainfall data as an explanatory variable
            sm = ps.StressModel(precip, ps.Gamma(), name="rainfall", up=True, settings="prec")  # up= True
            ml.add_stressmodel(sm)

            # Add the temperature data as an explanatory variable
            sm2 = ps.StressModel(temp, ps.Gamma(), up=False, name="temperature", settings='evap')
            ml.add_stressmodel(sm2)

            # Add the solar data as an explanatory variable
            sm3 = ps.StressModel(sol, ps.Gamma(), up=False, name="solar_radiation", settings='evap')
            ml.add_stressmodel(sm3)

            # Add the LAI data as an explanatory variable
            sm5 = ps.StressModel(lai, ps.Gamma(), up=True, name="LAI", settings='prec')
            ml.add_stressmodel(sm5)

            # Add the relative humidity data as an explanatory variable
            sm6 = ps.StressModel(hum, ps.Gamma(),up=True, name="relative_humidity", settings='prec')
            ml.add_stressmodel(sm6)

            # Add the wind speed data as an explanatory variable
            sm7= ps.StressModel(ws, ps.Gamma(),name="Wind_speed", settings='evap')
            ml.add_stressmodel(sm7)

            if( column==f'vwcdata5'):
                # Add the snow_depth data as an explanatory variable
                sm8= ps.StressModel(sd, ps.Gamma(), name="Snow_depth", settings='evap')
                ml.add_stressmodel(sm8)
            else:
                # Add the snow_depth data as an explanatory variable
                sm8= ps.StressModel(sd, ps.Gamma(), name="Snow_depth", settings='prec')
                ml.add_stressmodel(sm8)

            # Add the albedo data as an explanatory variable
            sm9= ps.StressModel(albedo, ps.Gamma(),up=False, name="albedo", settings='evap')
            ml.add_stressmodel(sm9)

            # Get tmin and tmax from the index of the series
            tmin = globals()[column].index[0]
            #tmax_solve = globals()[column].index[int(0.99*len(globals()[column]))]
            tmax_solve = globals()[column].index[-1]
            #tmax_plot = globals()[column].index[-1] + pd.Timedelta(days=days)
            tmax_plot = globals()[column].index[-1] 
            #print(tmax_solve,tmax_plot)

            # Solve the model
            ml.solve(tmin=tmin, tmax=tmax_solve)
                
            # Plot the results
            #ml.plot(tmax=tmax_plot)
            # Get the values used for plotting
            
            y_observed = ml.observations()
            y_predicted = ml.simulate(tmax=tmax_plot)  # predicted values

            # Trim y_predicted to match the length of y_observed
            y_predicted_trimmed = y_predicted[:len(y_observed)]

            # Calculate R^2 score
            r2 = r2_score(y_observed, y_predicted_trimmed)
            print('r2', series_name,':',r2)
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_observed, y_predicted_trimmed))
            trials += 1

        # # Print the RMSE value
        # print(f"RMSE_{column}:", rmse)

        # # # Print the R^2 score
        # print(f"R^2 score_{column}:", r2)

        # # # Print or use the values as needed
        
        # # #print("Observed Values:", y_observed)
        # # #print("Predicted Values:", y_predicted)

        # from matplotlib import rcParams
        # import matplotlib.pyplot as plt
        # # Creating a DataFrame for plotting
        # df_plot = pd.DataFrame({'Observed': y_observed, 'Predicted': y_predicted})
        # # Set Arial Bold as the default font for the plot
        # rcParams['font.family'] = 'Arial'
        # rcParams['font.weight'] = 'bold'
        # rcParams['font.size'] = 18


        # # Plotting
        # plt.figure(figsize=(12, 8))
        # plt.plot(df_plot.index, df_plot['Observed'], label='Measured', linewidth=2,color='black')
        # plt.plot(df_plot.index, df_plot['Predicted'], label='Modelled', linestyle='dashed', linewidth=2.5,color='brown')

        # # Customize plot
        # plt.xlabel('Date', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # if( column==f'ppdata1'):
        #     plt.ylabel('PWP (kPa)', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # else:
        #      plt.ylabel('VWC (%)', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # #plt.xticks(rotation=45, ha='right')
        # plt.xticks(df_plot.index[::90])  # Show every 90th date
        # plt.tick_params(axis='both', which='both', direction='in', labelsize=22)

        # # Add legend
        # plt.legend(fontsize=24)
        # # Save the plot with 600 dpi
        # plt.savefig(f'pastas_{column}.png', dpi=600)

        # # # Show the plot
        # plt.show()
        y_predicted.index = pd.to_datetime(y_predicted.index, utc=True).date
        #print(forecast_df_mean_per_day)

        # Merge the DataFrames based on the 'mid_time' column
        forecast_df_mean_per_day = pd.merge(forecast_df_mean_per_day, pd.DataFrame({f"{series_name}": y_predicted}), left_on='date', right_index=True, how='left')
        #print(forecast_df_mean_per_day)
    return forecast_df_mean_per_day

# In[6]:

def FoS_Predictions(final_result,Features_FoS,blob_client_loader):
    
    
    columns_to_predict = [ 'VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m', 'AirTemperature', 'LAI']
    # Extract relevant columns from 'final_result'    
    data_to_predict = final_result[columns_to_predict]

    best_regressor_rf = load_model(os.environ["REGRESSOR_FOS_9_PATH"], blob_client_loader)

    # Load the scaler
    scaler_rf = load_model(os.environ["SCALER_FOS_9_PATH"], blob_client_loader)
    
    # Scale the data using the loaded scaler
    scaled_data_to_predict_rf = scaler_rf.transform(data_to_predict) 
    # Make predictions using the loaded RF regressor
    fos_predictions_rf = best_regressor_rf.predict(scaled_data_to_predict_rf)  


    best_regressor_pr = load_model(os.environ["REGRESSOR_FOS_PR_PATH"], blob_client_loader)
    # Load the scaler
    scaler_pr = load_model(os.environ["SCALER_FOS_PR_PATH"], blob_client_loader) 
    converter_pr= load_model(os.environ["POLY_CONVERT_FOS_PR_PATH"], blob_client_loader)  
    
    # Scale the data using the loaded scaler
    scaled_data_to_predict_pr = scaler_pr.transform(data_to_predict)
    #convert data for PR
    converted_data_to_predict_pr=converter_pr.transform(scaled_data_to_predict_pr)    
    # Make predictions using the loaded RF regressor
    fos_predictions_pr = best_regressor_pr.predict(converted_data_to_predict_pr)
    
    # Add 'fos_predictions' as new columns to 'final_result'
    final_result['FoS_predictions'] = fos_predictions_rf
    final_result['FoS_predictions_PR'] = fos_predictions_pr

    # Save the result to a CSV file
    #last_mid_time = final_result['mid_time'].iloc[-1].strftime('%Y-%m-%dT%H-%M-%S')
   
    #file_name = f"FoS_prediction_{last_mid_time}.csv"
    #final_result.to_csv(file_name, index=False)

    # Print the resulting dataframe
    #print(final_result)

    return final_result