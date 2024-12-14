import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
LATITUDE = os.getenv('LATITUDE')
LONGITUDE = os.getenv('LONGITUDE')

# OpenWeatherMap Air Pollution API URLs

AIR_POLLUTION_HISTORY_URL = 'http://api.openweathermap.org/data/2.5/air_pollution/history'

# Directories

AIR_QUALITY_HISTORICAL_DIR = 'data/air_quality/historical/'

# Ensure directories exist
for directory in [AIR_QUALITY_HISTORICAL_DIR]:
    os.makedirs(directory, exist_ok=True)

def fetch_historical_air_quality(start_timestamp, end_timestamp):
    """
    Fetch historical air quality data between start_timestamp and end_timestamp.
    Timestamps should be in Unix format.
    """
    url = f'{AIR_POLLUTION_HISTORY_URL}?lat={LATITUDE}&lon={LONGITUDE}&start={start_timestamp}&end={end_timestamp}&appid={OPENWEATHER_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        filename = f'air_quality_historical.json'
        filepath = os.path.join(AIR_QUALITY_HISTORICAL_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'Historical Air Quality data saved to {filepath}')
    except Exception as e:
        print(f'Error fetching historical air quality data: {e}')

if __name__ == '__main__':

    fetch_historical_air_quality(1730419200, 1733011200)  # 2024-11-01 to 2024-12-01
