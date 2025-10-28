# Real-Time-Traffic-Prediction-System
# 🚗 Real-Time Traffic Prediction System

A comprehensive data science project that predicts traffic congestion levels using machine learning and deep learning models with real-time visualization.

## 🎯 Features

- **Real-time traffic data collection** from Google Maps and weather APIs
- **Multiple ML models**: Random Forest, Gradient Boosting, Logistic Regression, LSTM
- **Interactive dashboard** with live traffic map and predictions
- **Weather impact analysis** on traffic patterns
- **Time-series forecasting** for traffic congestion
- **RESTful API** for predictions

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Real-Time-Traffic-Prediction-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. **Get API Keys**
- [Google Maps API](https://developers.google.com/maps/documentation)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [TomTom API](https://developer.tomtom.com/) (optional)

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python main.py --mode full --samples 1000
```

### Option 2: Step-by-step Execution

1. **Collect Data**
```bash
python main.py --mode collect --samples 1000
```

2. **Train Models**
```bash
python main.py --mode train
```

3. **Launch Dashboard**
```bash
python main.py --mode dashboard
# Or directly: streamlit run dashboard.py
```

## 📊 Dashboard Features

- **Live Traffic Map**: Interactive map showing real-time congestion levels
- **Traffic Prediction**: Predict congestion for specific locations and times
- **Analytics**: Time-series plots, weather impact analysis, key metrics
- **Auto-refresh**: Real-time data updates every 30 seconds

## 🧠 Machine Learning Models

| Model | Type | Use Case |
|-------|------|----------|
| **Random Forest** | Ensemble | Feature importance analysis |
| **Gradient Boosting** | Ensemble | High accuracy predictions |
| **Logistic Regression** | Linear | Baseline model |
| **LSTM** | Deep Learning | Time-series forecasting |

## 📁 Project Structure

```
Real-Time-Traffic-Prediction-System/
├── config.py              # Configuration settings
├── data_collector.py      # Data collection from APIs
├── data_preprocessor.py   # Data cleaning and feature engineering
├── models.py             # ML/DL model implementations
├── dashboard.py          # Streamlit dashboard
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # Project documentation
```

## 🔧 Configuration

Edit `config.py` to customize:
- API endpoints and keys
- Model hyperparameters
- Database settings
- Congestion level thresholds

## 📈 Model Performance

The system automatically evaluates all models and selects the best performer:

- **Accuracy**: Classification accuracy on test set
- **Precision/Recall**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results

## 🌐 API Usage

### Predict Traffic Congestion
```python
from models import TrafficPredictionModels

# Load trained model
model = TrafficPredictionModels()
model.load_models()

# Make prediction
prediction = model.predict(input_data)
print(f"Congestion Level: {prediction}")
```

## 📊 Data Features

### Input Features
- **Location**: Latitude, Longitude
- **Time**: Hour, Day of week, Weekend flag
- **Weather**: Temperature, Humidity, Weather condition
- **Traffic**: Historical speed, Peak hour indicator

### Target Variable
- **Congestion Level**: 0-4 scale (Free Flow to Severe Congestion)

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are correctly set in `.env` file
   - Check API quotas and billing

2. **Model Training Errors**
   - Verify data file exists and has correct format
   - Check for sufficient data samples (minimum 100)

3. **Dashboard Not Loading**
   - Install streamlit: `pip install streamlit`
   - Run: `streamlit run dashboard.py`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Maps API for traffic data
- OpenWeatherMap for weather data
- Streamlit for dashboard framework
- TensorFlow/Keras for deep learning models

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Email: a9106072@gmail.com

---

**Happy Traffic Predicting! 🚗📊**
