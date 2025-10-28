#!/usr/bin/env python3
"""
Demo script for Real-Time Traffic Prediction System
Runs without external API dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simple configuration without external dependencies
CONGESTION_LEVELS = {
    0: 'Free Flow',
    1: 'Light Traffic', 
    2: 'Moderate Traffic',
    3: 'Heavy Traffic',
    4: 'Severe Congestion'
}

class SimpleTrafficPredictor:
    def __init__(self):
        self.model_weights = {
            'hour_weight': 0.3,
            'weather_weight': 0.2,
            'weekend_weight': 0.1,
            'location_weight': 0.4
        }
    
    def generate_sample_data(self, num_samples=100):
        """Generate sample traffic data"""
        np.random.seed(42)
        
        data = []
        for i in range(num_samples):
            # Generate timestamp
            base_time = datetime.now() - timedelta(days=7)
            timestamp = base_time + timedelta(minutes=np.random.randint(0, 10080))
            
            # Extract features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            
            # Generate location (NYC area)
            lat = 40.7128 + np.random.normal(0, 0.1)
            lon = -74.0060 + np.random.normal(0, 0.1)
            
            # Generate weather
            weather_conditions = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], p=[0.6, 0.25, 0.1, 0.05])
            temperature = np.random.normal(20, 10)
            
            # Calculate congestion based on rules
            congestion_score = 0
            
            # Time-based congestion
            if is_peak_hour:
                congestion_score += 2
            if hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Late night/early morning
                congestion_score -= 1
            
            # Weather impact
            if weather_conditions == 'Rain':
                congestion_score += 1
            elif weather_conditions in ['Snow', 'Fog']:
                congestion_score += 2
            
            # Weekend effect
            if is_weekend:
                congestion_score -= 1
            
            # Random variation
            congestion_score += np.random.normal(0, 0.5)
            
            # Convert to congestion level (0-4)
            congestion_level = max(0, min(4, int(congestion_score + 2)))
            
            data.append({
                'timestamp': timestamp,
                'lat': lat,
                'lon': lon,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour,
                'temperature': temperature,
                'weather_condition': weather_conditions,
                'congestion_level': congestion_level
            })
        
        return pd.DataFrame(data)
    
    def predict_congestion(self, hour, day_of_week, weather, temperature):
        """Simple rule-based prediction"""
        score = 2  # Base score
        
        # Time factors
        if hour in [7, 8, 9, 17, 18, 19]:  # Peak hours
            score += 1.5
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Off-peak
            score -= 1
        
        # Day factors
        if day_of_week >= 5:  # Weekend
            score -= 0.5
        
        # Weather factors
        if weather == 'Rain':
            score += 0.8
        elif weather in ['Snow', 'Fog']:
            score += 1.5
        
        # Temperature factors
        if temperature < 0 or temperature > 35:
            score += 0.5
        
        return max(0, min(4, int(score)))

def run_demo():
    """Run traffic prediction demo"""
    print("Real-Time Traffic Prediction System Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SimpleTrafficPredictor()
    
    # Generate sample data
    print("Generating sample traffic data...")
    df = predictor.generate_sample_data(500)
    print(f"Generated {len(df)} traffic records")
    
    # Show data summary
    print("\nData Summary:")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Location range: Lat {df['lat'].min():.3f} to {df['lat'].max():.3f}")
    print(f"                Lon {df['lon'].min():.3f} to {df['lon'].max():.3f}")
    
    # Show congestion distribution
    print("\nCongestion Level Distribution:")
    congestion_counts = df['congestion_level'].value_counts().sort_index()
    for level, count in congestion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {CONGESTION_LEVELS[level]:18}: {count:3d} ({percentage:5.1f}%)")
    
    # Show peak hour analysis
    print("\nPeak Hour Analysis:")
    peak_congestion = df[df['is_peak_hour'] == 1]['congestion_level'].mean()
    off_peak_congestion = df[df['is_peak_hour'] == 0]['congestion_level'].mean()
    print(f"  Peak hours average congestion:     {peak_congestion:.2f}")
    print(f"  Off-peak hours average congestion: {off_peak_congestion:.2f}")
    
    # Show weather impact
    print("\nWeather Impact Analysis:")
    weather_impact = df.groupby('weather_condition')['congestion_level'].mean().sort_values(ascending=False)
    for weather, avg_congestion in weather_impact.items():
        print(f"  {weather:10}: {avg_congestion:.2f} average congestion")
    
    # Interactive prediction
    print("\nTraffic Prediction Examples:")
    test_cases = [
        (8, 1, 'Clear', 20),      # Monday 8 AM, Clear weather
        (18, 4, 'Rain', 15),      # Friday 6 PM, Rainy
        (14, 6, 'Clear', 25),     # Saturday 2 PM, Clear
        (2, 2, 'Snow', -5),       # Tuesday 2 AM, Snowy
    ]
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for hour, day_of_week, weather, temp in test_cases:
        prediction = predictor.predict_congestion(hour, day_of_week, weather, temp)
        day_name = days[day_of_week]
        
        print(f"  {day_name} {hour:2d}:00, {weather:5}, {temp:3d}C -> {CONGESTION_LEVELS[prediction]}")
    
    # Save sample data
    df.to_csv('demo_traffic_data.csv', index=False)
    print(f"\nSample data saved to 'demo_traffic_data.csv'")
    
    print("\nDemo completed successfully!")
    print("\nNext steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Set up API keys in .env file")
    print("3. Run full system: python main.py --mode full")
    print("4. Launch dashboard: streamlit run dashboard.py")

if __name__ == "__main__":
    run_demo()