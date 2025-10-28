import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from config import GOOGLE_MAPS_API_KEY, OPENWEATHER_API_KEY

class TrafficDataCollector:
    def __init__(self):
        self.google_api_key = GOOGLE_MAPS_API_KEY
        self.weather_api_key = OPENWEATHER_API_KEY
        
    def get_traffic_data(self, origin, destination):
        """Collect traffic data from Google Maps API"""
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': origin,
            'destination': destination,
            'departure_time': 'now',
            'traffic_model': 'best_guess',
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                route = data['routes'][0]['legs'][0]
                return {
                    'distance': route['distance']['value'],
                    'duration': route['duration']['value'],
                    'duration_in_traffic': route.get('duration_in_traffic', {}).get('value', route['duration']['value']),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error collecting traffic data: {e}")
        return None
    
    def get_weather_data(self, lat, lon):
        """Collect weather data from OpenWeatherMap API"""
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.weather_api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather_condition': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed'],
                'visibility': data.get('visibility', 10000),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error collecting weather data: {e}")
        return None
    
    def generate_sample_data(self, num_samples=1000):
        """Generate sample traffic data for testing"""
        np.random.seed(42)
        
        # Define major routes in a city
        routes = [
            {'origin': (40.7128, -74.0060), 'destination': (40.7589, -73.9851)},  # NYC example
            {'origin': (40.7589, -73.9851), 'destination': (40.6892, -74.0445)},
            {'origin': (40.6892, -74.0445), 'destination': (40.7128, -74.0060)},
        ]
        
        data = []
        for i in range(num_samples):
            route = np.random.choice(routes)
            
            # Generate timestamp
            base_time = datetime.now() - timedelta(days=30)
            timestamp = base_time + timedelta(minutes=np.random.randint(0, 43200))
            
            # Extract time features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            
            # Generate traffic features
            base_speed = 50 + np.random.normal(0, 10)
            if is_peak_hour:
                base_speed *= 0.6
            if is_weekend:
                base_speed *= 1.2
                
            # Weather impact
            weather_conditions = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], p=[0.6, 0.25, 0.1, 0.05])
            if weather_conditions == 'Rain':
                base_speed *= 0.8
            elif weather_conditions == 'Snow':
                base_speed *= 0.6
            elif weather_conditions == 'Fog':
                base_speed *= 0.7
            
            # Calculate congestion level
            if base_speed > 45:
                congestion_level = 0  # Free Flow
            elif base_speed > 35:
                congestion_level = 1  # Light Traffic
            elif base_speed > 25:
                congestion_level = 2  # Moderate Traffic
            elif base_speed > 15:
                congestion_level = 3  # Heavy Traffic
            else:
                congestion_level = 4  # Severe Congestion
            
            data.append({
                'lat': route['origin'][0] + np.random.normal(0, 0.01),
                'lon': route['origin'][1] + np.random.normal(0, 0.01),
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour,
                'temperature': np.random.normal(20, 10),
                'humidity': np.random.uniform(30, 90),
                'weather_condition': weather_conditions,
                'wind_speed': np.random.uniform(0, 20),
                'average_speed': max(5, base_speed),
                'congestion_level': congestion_level
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    collector = TrafficDataCollector()
    sample_data = collector.generate_sample_data(1000)
    sample_data.to_csv('sample_traffic_data.csv', index=False)
    print("Sample data generated and saved to sample_traffic_data.csv")