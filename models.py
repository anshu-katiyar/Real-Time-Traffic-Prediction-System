import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class TrafficPredictionModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        return gb_model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        lr_model = LogisticRegression(
            multi_class='ovr',
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        return lr_model
    
    def create_lstm_model(self, input_shape, num_classes):
        """Create LSTM model for time series prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_data(self, X, y, sequence_length=10):
        """Prepare data for LSTM model"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_lstm(self, X_train, y_train, sequence_length=10, epochs=50):
        """Train LSTM model"""
        # Prepare sequential data
        X_seq, y_seq = self.prepare_lstm_data(X_train, y_train, sequence_length)
        
        if len(X_seq) == 0:
            print("Not enough data for LSTM training")
            return None
        
        # Create and train model
        lstm_model = self.create_lstm_model(
            input_shape=(sequence_length, X_train.shape[1]),
            num_classes=len(np.unique(y_train))
        )
        
        history = lstm_model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['lstm'] = lstm_model
        return lstm_model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        if model_name == 'lstm':
            # For LSTM, prepare sequential test data
            X_seq, y_seq = self.prepare_lstm_data(X_test, y_test, 10)
            if len(X_seq) == 0:
                return None
            y_pred = model.predict(X_seq)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y_seq
        else:
            y_pred = model.predict(X_test)
            y_true = y_test
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{model_name.upper()} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return accuracy
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        results = {}
        
        # Train traditional ML models
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        rf_accuracy = self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        results['random_forest'] = rf_accuracy
        
        print("\nTraining Gradient Boosting...")
        gb_model = self.train_gradient_boosting(X_train, y_train)
        gb_accuracy = self.evaluate_model(gb_model, X_test, y_test, 'gradient_boosting')
        results['gradient_boosting'] = gb_accuracy
        
        print("\nTraining Logistic Regression...")
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_accuracy = self.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
        results['logistic_regression'] = lr_accuracy
        
        # Train LSTM model if enough data
        if len(X_train) > 50:
            print("\nTraining LSTM...")
            lstm_model, history = self.train_lstm(X_train, y_train, epochs=30)
            if lstm_model:
                lstm_accuracy = self.evaluate_model(lstm_model, X_test, y_test, 'lstm')
                results['lstm'] = lstm_accuracy
        
        # Find best model
        self.best_model = max(results, key=results.get)
        self.best_score = results[self.best_model]
        
        print(f"\nBest Model: {self.best_model} (Accuracy: {self.best_score:.4f})")
        
        return results
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            # For LSTM, need sequential data
            if len(X) < 10:
                # If not enough data, use the last available prediction
                return np.array([2])  # Default to moderate traffic
            X_seq = X.iloc[-10:].values.reshape(1, 10, -1)
            predictions = model.predict(X_seq)
            return np.argmax(predictions, axis=1)
        else:
            return model.predict(X)
    
    def save_models(self, filepath_prefix='traffic_model'):
        """Save trained models"""
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(f'{filepath_prefix}_{name}.h5')
            else:
                joblib.dump(model, f'{filepath_prefix}_{name}.pkl')
        
        # Save model metadata
        metadata = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'available_models': list(self.models.keys())
        }
        joblib.dump(metadata, f'{filepath_prefix}_metadata.pkl')
    
    def load_models(self, filepath_prefix='traffic_model'):
        """Load trained models"""
        metadata = joblib.load(f'{filepath_prefix}_metadata.pkl')
        self.best_model = metadata['best_model']
        self.best_score = metadata['best_score']
        
        for model_name in metadata['available_models']:
            if model_name == 'lstm':
                self.models[model_name] = tf.keras.models.load_model(f'{filepath_prefix}_{model_name}.h5')
            else:
                self.models[model_name] = joblib.load(f'{filepath_prefix}_{model_name}.pkl')

if __name__ == "__main__":
    # Example usage
    from data_preprocessor import TrafficDataPreprocessor
    
    preprocessor = TrafficDataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline('sample_traffic_data.csv')
    
    # Train models
    model_trainer = TrafficPredictionModels()
    results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save models
    model_trainer.save_models()