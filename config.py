import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'your_google_maps_api_key')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'your_openweather_api_key')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY', 'your_tomtom_api_key')

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'traffic_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Model Parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'lstm_units': 50,
    'epochs': 100,
    'batch_size': 32
}

# Traffic Congestion Levels
CONGESTION_LEVELS = {
    0: 'Free Flow',
    1: 'Light Traffic',
    2: 'Moderate Traffic',
    3: 'Heavy Traffic',
    4: 'Severe Congestion'
}