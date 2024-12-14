# scripts/weather_collector.py

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VISUAL_CROSSING_API_KEY = os.getenv('VISUAL_CROSSING_API_KEY')
LATITUDE = os.getenv('LATITUDE')
LONGITUDE = os.getenv('LONGITUDE')
LOCATION = f"{LATITUDE},{LONGITUDE}"


# Directories
WEATHER_HISTORICAL_DIR = 'data/weather/historical/'

# Ensure directories exist
for directory in [WEATHER_HISTORICAL_DIR]:
    os.makedirs(directory, exist_ok=True)

def fetch_historical_weather(start_date, end_date):
    """
    Fetch historical weather data between start_date and end_date.
    Dates should be in 'YYYY-MM-DD' format.
    """
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}/{start_date}/{end_date}?unitGroup=metric&key={VISUAL_CROSSING_API_KEY}&include=days,hours&timezone=Z'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        filename = f'weather_historical.json'
        filepath = os.path.join(WEATHER_HISTORICAL_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'Historical Weather data saved to {filepath}')
    except Exception as e:
        print(f'Error fetching historical weather data: {e}')

if __name__ == '__main__':
    # Example: Fetch historical data from November 1, 2024, to December 1, 2024
    fetch_historical_weather('2024-11-01', '2024-12-01')
