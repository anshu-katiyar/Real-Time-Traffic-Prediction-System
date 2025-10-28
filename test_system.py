#!/usr/bin/env python3
"""
Test script for Real-Time Traffic Prediction System
"""

import pandas as pd
import numpy as np
from data_collector import TrafficDataCollector
from data_preprocessor import TrafficDataPreprocessor
from models import TrafficPredictionModels

def test_data_collection():
    """Test data collection functionality"""
    print("Testing data collection...")
    collector = TrafficDataCollector()
    
    # Generate sample data
    data = collector.generate_sample_data(100)
    
    assert len(data) == 100, "Data collection failed"
    assert 'congestion_level' in data.columns, "Missing congestion_level column"
    assert data['congestion_level'].min() >= 0, "Invalid congestion level"
    assert data['congestion_level'].max() <= 4, "Invalid congestion level"
    
    print("✅ Data collection test passed")
    return data

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("Testing data preprocessing...")
    
    # Generate test data
    collector = TrafficDataCollector()
    data = collector.generate_sample_data(200)
    data.to_csv('test_data.csv', index=False)
    
    # Preprocess data
    preprocessor = TrafficDataPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.preprocess_pipeline('test_data.csv')
    
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"
    assert X_train.shape[1] > 10, "Insufficient features"
    
    print("✅ Data preprocessing test passed")
    return X_train, X_test, y_train, y_test

def test_model_training():
    """Test model training functionality"""
    print("Testing model training...")
    
    # Get preprocessed data
    X_train, X_test, y_train, y_test = test_data_preprocessing()
    
    # Train models
    model_trainer = TrafficPredictionModels()
    
    # Test individual models
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    assert rf_model is not None, "Random Forest training failed"
    
    gb_model = model_trainer.train_gradient_boosting(X_train, y_train)
    assert gb_model is not None, "Gradient Boosting training failed"
    
    lr_model = model_trainer.train_logistic_regression(X_train, y_train)
    assert lr_model is not None, "Logistic Regression training failed"
    
    # Test predictions
    predictions = rf_model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length mismatch"
    assert all(0 <= p <= 4 for p in predictions), "Invalid prediction values"
    
    print("✅ Model training test passed")
    return model_trainer

def test_prediction():
    """Test prediction functionality"""
    print("Testing prediction functionality...")
    
    # Train a simple model
    model_trainer = test_model_training()
    
    # Create test input
    test_input = pd.DataFrame({
        'lat': [40.7128],
        'lon': [-74.0060],
        'hour': [8],
        'day_of_week': [1],
        'is_weekend': [0],
        'is_peak_hour': [1],
        'temperature': [20],
        'humidity': [60],
        'wind_speed': [10],
        'hour_sin': [np.sin(2 * np.pi * 8 / 24)],
        'hour_cos': [np.cos(2 * np.pi * 8 / 24)],
        'day_sin': [np.sin(2 * np.pi * 1 / 7)],
        'day_cos': [np.cos(2 * np.pi * 1 / 7)],
        'weather_impact': [1.0],
        'is_rush_hour': [1],
        'weather_condition_encoded': [0]
    })
    
    # Make prediction
    prediction = model_trainer.predict(test_input, 'random_forest')
    
    assert len(prediction) == 1, "Single prediction failed"
    assert 0 <= prediction[0] <= 4, "Invalid prediction value"
    
    print("✅ Prediction test passed")

def run_all_tests():
    """Run all system tests"""
    print("🚗 Running Real-Time Traffic Prediction System Tests")
    print("=" * 60)
    
    try:
        # Test data collection
        test_data_collection()
        
        # Test data preprocessing
        test_data_preprocessing()
        
        # Test model training
        test_model_training()
        
        # Test prediction
        test_prediction()
        
        print("\n🎉 All tests passed successfully!")
        print("The Real-Time Traffic Prediction System is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise e

if __name__ == "__main__":
    run_all_tests()