import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class TrafficDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load traffic data from CSV file"""
        return pd.read_csv(file_path)
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['temperature'].fillna(df['temperature'].mean(), inplace=True)
        df['humidity'].fillna(df['humidity'].mean(), inplace=True)
        df['wind_speed'].fillna(df['wind_speed'].mean(), inplace=True)
        
        # Remove outliers using IQR method
        numeric_columns = ['temperature', 'humidity', 'wind_speed', 'average_speed']
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def feature_engineering(self, df):
        """Create additional features"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Create cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weather impact score
        weather_impact = {
            'Clear': 1.0,
            'Rain': 0.8,
            'Snow': 0.6,
            'Fog': 0.7
        }
        df['weather_impact'] = df['weather_condition'].map(weather_impact)
        
        # Rush hour indicator
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_columns = ['weather_condition']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """Normalize numerical features"""
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df_normalized
    
    def prepare_features_target(self, df):
        """Prepare feature matrix and target variable"""
        feature_columns = [
            'lat', 'lon', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
            'temperature', 'humidity', 'wind_speed', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'weather_impact', 'is_rush_hour',
            'weather_condition_encoded'
        ]
        
        X = df[feature_columns]
        y = df['congestion_level']
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def preprocess_pipeline(self, file_path):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(file_path)
        print(f"Loaded {len(df)} records")
        
        # Clean data
        df = self.clean_data(df)
        print(f"After cleaning: {len(df)} records")
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Normalize features
        X_train_normalized = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_normalized = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_normalized, X_test_normalized, y_train, y_test, df

if __name__ == "__main__":
    preprocessor = TrafficDataPreprocessor()
    
    # Generate sample data first
    from data_collector import TrafficDataCollector
    collector = TrafficDataCollector()
    sample_data = collector.generate_sample_data(1000)
    sample_data.to_csv('sample_traffic_data.csv', index=False)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, processed_df = preprocessor.preprocess_pipeline('sample_traffic_data.csv')
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target distribution:\n{y_train.value_counts().sort_index()}")