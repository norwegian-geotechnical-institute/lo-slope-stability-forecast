import requests
import os
import pandas as pd
from io import StringIO
from datetime import datetime, timezone


def get_access_token(token_provider_url: str, client_id: str, client_secret: str):
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    r = requests.post(token_provider_url, data=data)
    if not r.ok:
        raise Exception(f"Token request failed with error code {r.status_code}: {r.reason}")
    response_data = r.json()
    return response_data["access_token"]


# Frost API wants times in UTC, but otherwise iso-8601
# Ref.: https://frost.met.no/concepts2.html#timespecifications
def convert_time_for_frost_api(timestamp: datetime):
    return timestamp.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def fetch_from_ngi_live(project_id, start_time, end_time, logger_name, sensor_type, secret_client):
    payload = {
        "sensor_logger": {
            "project": project_id,
            "logger": logger_name,
            "sensor_type": sensor_type
        },
        "sample": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "status_code": 0,
            "resample": {
                "method": "pre_mean",
                "interval": "1 day"
            }
        }
    }
    access_token = get_access_token(
        os.environ["NGILIVE_API_TOKEN_PROVIDER_URL"],
        secret_client.get_secret(os.environ['NGILIVE_API_CLIENT_ID_SECRET']),
        secret_client.get_secret(os.environ['NGILIVE_API_CLIENT_SECRET_SECRET'])
    )
    r = requests.post(
        f"{os.environ['NGILIVE_API_URL']}/datapoints/logger",
        json=payload,
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if not r.ok:
        raise Exception(f"Request to ngi live API failed with error code {r.status_code}: {r.reason}")

    csvStringIO = StringIO(r.text)
    return pd.read_csv(csvStringIO, sep=",")

"""
Created on Sunday February 5 14:30:00 2023

@author: EAO
"""

def GetWeatherForecast():
    """
    This function accesses the weather forecast data from the Norwegian Meteorological Institute, and returns the required data to the user.
    Written by: Emir Ahmet Oguz, February 5 14:30:00 2023

    Returns
    -------
    Time : list
        List of times in the forecast data as string.
    Duration : list
        List of duration of the forrecast at aach time as float.
    Relative humidity: list
        List of relative humidity in the forecast data as float.
    AirTemperature : list
        List of air temperature in the forecast data as float.
    Precipitation : list
        List of precipitation in the forecast data as float.

    """

    ## Libraries
    ## https://docs.python.org/3/library/urllib.request.html#module-urllib.request
    ## https://docs.python.org/3/library/json.html
    import urllib, json
    from urllib import request, error

    '''
    ## For the following section, "https://docs.python.org/3/howto/urllib2.html" is utilized.
    '''
    ## Define url including the coordinates of the desired location
    ## Municipality of Eidsvoll, Norway (60°19′23.376", 11°14′44.646")
    # Latitude  = 60.322
    # Longitude = 11.245    ## https://www.latlong.net/  & https://www.google.com/maps/search/church+near+Eidsvoll+Municipality/@60.3236357,11.2462548,542m/data=!3m1!1e3
    # Altitude  = 170       ## https://en-gb.topographic-map.com/map-m45k/Norway/?center=60.32248%2C11.24637&zoom=16&base=6&popup=60.3223%2C11.24617
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=60.322&lon=11.245&altitude=170"
    ## https://www.whatismybrowser.com/detect/what-is-my-user-agent/
    User_Agent = os.environ['MET_API_USER_AGENT']

    ## Define data to send to the server
    Values = {'name': 'NGI',
                'location': 'Eidsvoll,',
                'language': 'Python' }
    Headers = {'User-Agent': User_Agent}

    ## Generate requert
    Data = urllib.parse.urlencode(Values)
    Data = Data.encode('ascii')
    #req = request.Request(url, Data, Headers)
    headers = {'user-agent': User_Agent}
    r6 = requests.get(url, headers=headers)
    if r6.ok:
            # Extract JSON data
        Data = r6.json()
    else:
        print (r6.reason)
        return
    #print(Data)


    ## Open and read url
    # with urllib.request.urlopen(req) as response:
    #    The_page = response.read()
    '''
    '''
    ## Take the page and returns the json object
    # Data = json.loads(The_page)
    #print(Data)

    ## Allocate lists for required information: Time, air temperature (celcius), precipitation (mm) and duration (hour)
    Time             = []
    Duration         = []
    RelativeHumidity = []
    AirTemperature   = []
    Precipitation    = []
    wind_speed       = []

    ## Read required values from the data
    for Pred in Data['properties']['timeseries']:

        ## Time current prediction
        Time.append(Pred['time'])
        AirTemperature.append(Pred['data']['instant']['details']['air_temperature'])
        RelativeHumidity.append(Pred['data']['instant']['details']['relative_humidity'])
        wind_speed.append(Pred['data']['instant']['details']['wind_speed'])


        ## Next 1 hour precipitation prediction
        if('next_1_hours' in Pred['data']):
            Precipitation.append(Pred['data']['next_1_hours']['details']['precipitation_amount'])
            Duration.append(1.0)
        ## Next 6 hour precipitation prediction (take if if there is no 1-hour prediction)
        elif('next_6_hours' in Pred['data']):
                Precipitation.append(Pred['data']['next_6_hours']['details']['precipitation_amount'])
                Duration.append(6.0)
        ## Next 12 hour precipitation prediction (take it if there is no 1 or 6 hour prediction)
        elif('next_12_hours' in Pred['data']):
                Precipitation.append(Pred['data']['next_12_hours']['details']['precipitation_amount'])
                Duration.append(12.0)

    ## Return information
    return Time, Duration, RelativeHumidity, AirTemperature, Precipitation, wind_speed