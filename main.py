#!/usr/bin/env python3
"""
Real-Time Traffic Prediction System
Main execution script for training models and running the system
"""

import argparse
import sys
from data_collector import TrafficDataCollector
from data_preprocessor import TrafficDataPreprocessor
from models import TrafficPredictionModels
from dashboard import TrafficDashboard

def collect_data(num_samples=1000):
    """Collect and save traffic data"""
    print("Collecting traffic data...")
    collector = TrafficDataCollector()
    
    # Generate sample data
    data = collector.generate_sample_data(num_samples)
    data.to_csv('traffic_data.csv', index=False)
    
    print(f"Collected {len(data)} traffic records and saved to traffic_data.csv")
    return data

def preprocess_data(data_file='traffic_data.csv'):
    """Preprocess traffic data"""
    print("Preprocessing traffic data...")
    preprocessor = TrafficDataPreprocessor()
    
    X_train, X_test, y_train, y_test, processed_df = preprocessor.preprocess_pipeline(data_file)
    
    print(f"Data preprocessed successfully")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, X_test, y_train, y_test):
    """Train all machine learning models"""
    print("Training machine learning models...")
    
    model_trainer = TrafficPredictionModels()
    results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save trained models
    model_trainer.save_models('traffic_model')
    
    print("Models trained and saved successfully")
    print(f"   Best model: {model_trainer.best_model}")
    print(f"   Best accuracy: {model_trainer.best_score:.4f}")
    
    return model_trainer, results

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("Launching traffic prediction dashboard...")
    dashboard = TrafficDashboard()
    dashboard.run_dashboard()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Real-Time Traffic Prediction System')
    parser.add_argument('--mode', choices=['collect', 'train', 'dashboard', 'full'], 
                       default='full', help='Operation mode')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Number of samples to generate')
    parser.add_argument('--data-file', default='traffic_data.csv', 
                       help='Data file path')
    
    args = parser.parse_args()
    
    print("Real-Time Traffic Prediction System")
    print("=" * 50)
    
    if args.mode in ['collect', 'full']:
        # Collect data
        collect_data(args.samples)
    
    if args.mode in ['train', 'full']:
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(args.data_file)
        
        # Train models
        model_trainer, results = train_models(X_train, X_test, y_train, y_test)
        
        print("\nModel Performance Summary:")
        print("-" * 30)
        for model_name, accuracy in results.items():
            print(f"{model_name:20}: {accuracy:.4f}")
    
    if args.mode in ['dashboard', 'full']:
        # Run dashboard
        print("\nStarting dashboard...")
        print("Open your browser and go to: http://localhost:8501")
        run_dashboard()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)