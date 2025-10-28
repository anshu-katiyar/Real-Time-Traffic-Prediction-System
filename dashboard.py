import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import requests

from data_collector import TrafficDataCollector
from data_preprocessor import TrafficDataPreprocessor
from models import TrafficPredictionModels
from config import CONGESTION_LEVELS

class TrafficDashboard:
    def __init__(self):
        self.collector = TrafficDataCollector()
        self.preprocessor = TrafficDataPreprocessor()
        self.models = TrafficPredictionModels()
        
        # Location data
        self.location_data = self.get_location_data()
        
        # Load models if available
        try:
            self.models.load_models()
        except:
            st.warning("No pre-trained models found. Please train models first.")
    
    def get_location_data(self):
        """Get hierarchical location data with all Indian states and major districts"""
        return {
            'India': {
                'Andhra Pradesh': {
                    'Visakhapatnam': [(17.6868, 83.2185)], 'Vijayawada': [(16.5062, 80.6480)], 
                    'Guntur': [(16.3067, 80.4365)], 'Nellore': [(14.4426, 79.9865)]
                },
                'Arunachal Pradesh': {
                    'Itanagar': [(27.0844, 93.6053)], 'Naharlagun': [(27.1045, 93.6967)]
                },
                'Assam': {
                    'Guwahati': [(26.1445, 91.7362)], 'Silchar': [(24.8333, 92.7789)], 
                    'Dibrugarh': [(27.4728, 94.9120)], 'Jorhat': [(26.7509, 94.2037)]
                },
                'Bihar': {
                    'Patna': [(25.5941, 85.1376)], 'Gaya': [(24.7914, 85.0002)], 
                    'Bhagalpur': [(25.2425, 86.9842)], 'Muzaffarpur': [(26.1209, 85.3647)]
                },
                'Chhattisgarh': {
                    'Raipur': [(21.2514, 81.6296)], 'Bhilai': [(21.1938, 81.3509)], 
                    'Korba': [(22.3595, 82.7501)], 'Bilaspur': [(22.0797, 82.1409)]
                },
                'Delhi': {
                    'New Delhi': [(28.6139, 77.2090)], 'Central Delhi': [(28.6562, 77.2410)], 
                    'South Delhi': [(28.5355, 77.2490)], 'North Delhi': [(28.7041, 77.1025)],
                    'East Delhi': [(28.6508, 77.2773)], 'West Delhi': [(28.6692, 77.1100)]
                },
                'Goa': {
                    'Panaji': [(15.4909, 73.8278)], 'Margao': [(15.2700, 73.9500)], 
                    'Vasco da Gama': [(15.3955, 73.8157)]
                },
                'Gujarat': {
                    'Ahmedabad': [(23.0225, 72.5714)], 'Surat': [(21.1702, 72.8311)], 
                    'Vadodara': [(22.3072, 73.1812)], 'Rajkot': [(22.3039, 70.8022)],
                    'Gandhinagar': [(23.2156, 72.6369)], 'Bhavnagar': [(21.7645, 72.1519)]
                },
                'Haryana': {
                    'Gurugram': [(28.4595, 77.0266)], 'Faridabad': [(28.4089, 77.3178)], 
                    'Panipat': [(29.3909, 76.9635)], 'Ambala': [(30.3782, 76.7767)],
                    'Karnal': [(29.6857, 76.9905)], 'Hisar': [(29.1492, 75.7217)]
                },
                'Himachal Pradesh': {
                    'Shimla': [(31.1048, 77.1734)], 'Dharamshala': [(32.2190, 76.3234)], 
                    'Manali': [(32.2396, 77.1887)], 'Kullu': [(31.9578, 77.1092)]
                },
                'Jharkhand': {
                    'Ranchi': [(23.3441, 85.3096)], 'Jamshedpur': [(22.8046, 86.2029)], 
                    'Dhanbad': [(23.7957, 86.4304)], 'Bokaro': [(23.6693, 85.9590)]
                },
                'Karnataka': {
                    'Bangalore': [(12.9716, 77.5946)], 'Mysore': [(12.2958, 76.6394)], 
                    'Hubli': [(15.3647, 75.1240)], 'Mangalore': [(12.9141, 74.8560)],
                    'Belgaum': [(15.8497, 74.4977)], 'Gulbarga': [(17.3297, 76.8343)]
                },
                'Kerala': {
                    'Thiruvananthapuram': [(8.5241, 76.9366)], 'Kochi': [(9.9312, 76.2673)], 
                    'Kozhikode': [(11.2588, 75.7804)], 'Thrissur': [(10.5276, 76.2144)],
                    'Kollam': [(8.8932, 76.6141)], 'Kannur': [(11.8745, 75.3704)]
                },
                'Madhya Pradesh': {
                    'Bhopal': [(23.2599, 77.4126)], 'Indore': [(22.7196, 75.8577)], 
                    'Gwalior': [(26.2183, 78.1828)], 'Jabalpur': [(23.1815, 79.9864)],
                    'Ujjain': [(23.1765, 75.7885)], 'Sagar': [(23.8388, 78.7378)]
                },
                'Maharashtra': {
                    'Mumbai': [(19.0760, 72.8777)], 'Pune': [(18.5204, 73.8567)], 
                    'Nagpur': [(21.1458, 79.0882)], 'Nashik': [(19.9975, 73.7898)],
                    'Aurangabad': [(19.8762, 75.3433)], 'Solapur': [(17.6599, 75.9064)],
                    'Thane': [(19.2183, 72.9781)], 'Kolhapur': [(16.7050, 74.2433)]
                },
                'Manipur': {
                    'Imphal': [(24.8170, 93.9368)], 'Thoubal': [(24.6340, 93.9896)]
                },
                'Meghalaya': {
                    'Shillong': [(25.5788, 91.8933)], 'Tura': [(25.5138, 90.2035)]
                },
                'Mizoram': {
                    'Aizawl': [(23.7271, 92.7176)], 'Lunglei': [(22.8774, 92.7348)]
                },
                'Nagaland': {
                    'Kohima': [(25.6751, 94.1086)], 'Dimapur': [(25.9044, 93.7267)]
                },
                'Odisha': {
                    'Bhubaneswar': [(20.2961, 85.8245)], 'Cuttack': [(20.4625, 85.8828)], 
                    'Rourkela': [(22.2604, 84.8536)], 'Berhampur': [(19.3149, 84.7941)]
                },
                'Punjab': {
                    'Chandigarh': [(30.7333, 76.7794)], 'Ludhiana': [(30.9010, 75.8573)], 
                    'Amritsar': [(31.6340, 74.8723)], 'Jalandhar': [(31.3260, 75.5762)],
                    'Patiala': [(30.3398, 76.3869)], 'Bathinda': [(30.2110, 74.9455)]
                },
                'Rajasthan': {
                    'Jaipur': [(26.9124, 75.7873)], 'Jodhpur': [(26.2389, 73.0243)], 
                    'Udaipur': [(24.5854, 73.7125)], 'Kota': [(25.2138, 75.8648)],
                    'Bikaner': [(28.0229, 73.3119)], 'Ajmer': [(26.4499, 74.6399)]
                },
                'Sikkim': {
                    'Gangtok': [(27.3389, 88.6065)], 'Namchi': [(27.1668, 88.3639)]
                },
                'Tamil Nadu': {
                    'Chennai': [(13.0827, 80.2707)], 'Coimbatore': [(11.0168, 76.9558)], 
                    'Madurai': [(9.9252, 78.1198)], 'Tiruchirappalli': [(10.7905, 78.7047)],
                    'Salem': [(11.6643, 78.1460)], 'Tirunelveli': [(8.7139, 77.7567)]
                },
                'Telangana': {
                    'Hyderabad': [(17.3850, 78.4867)], 'Warangal': [(17.9689, 79.5941)], 
                    'Nizamabad': [(18.6725, 78.0941)], 'Karimnagar': [(18.4386, 79.1288)]
                },
                'Tripura': {
                    'Agartala': [(23.8315, 91.2868)], 'Dharmanagar': [(24.3667, 92.1667)]
                },
                'Uttar Pradesh': {
                    'Lucknow': [(26.8467, 80.9462)], 'Kanpur': [(26.4499, 80.3319)], 
                    'Ghaziabad': [(28.6692, 77.4538)], 'Agra': [(27.1767, 78.0081)],
                    'Meerut': [(28.9845, 77.7064)], 'Varanasi': [(25.3176, 82.9739)],
                    'Allahabad': [(25.4358, 81.8463)], 'Bareilly': [(28.3670, 79.4304)]
                },
                'Uttarakhand': {
                    'Dehradun': [(30.3165, 78.0322)], 'Haridwar': [(29.9457, 78.1642)], 
                    'Nainital': [(29.3803, 79.4636)], 'Rishikesh': [(30.0869, 78.2676)]
                },
                'West Bengal': {
                    'Kolkata': [(22.5726, 88.3639)], 'Howrah': [(22.5958, 88.2636)], 
                    'Durgapur': [(23.4820, 87.3119)], 'Asansol': [(23.6739, 86.9524)],
                    'Siliguri': [(26.7271, 88.3953)], 'Malda': [(25.0961, 88.1408)]
                }
            },
            'USA': {
                'New York': {
                    'Manhattan': [(40.7831, -73.9712)], 'Brooklyn': [(40.6782, -73.9442)], 
                    'Queens': [(40.7282, -73.7949)], 'Bronx': [(40.8448, -73.8648)]
                },
                'California': {
                    'Los Angeles': [(34.0522, -118.2437)], 'San Francisco': [(37.7749, -122.4194)], 
                    'San Diego': [(32.7157, -117.1611)], 'Sacramento': [(38.5816, -121.4944)]
                }
            },
            'UK': {
                'England': {
                    'London': [(51.5074, -0.1278)], 'Manchester': [(53.4808, -2.2426)], 
                    'Birmingham': [(52.4862, -1.8904)], 'Liverpool': [(53.4084, -2.9916)]
                }
            }
        }
    
    def get_coordinates_for_location(self, country, state, district):
        """Get coordinates for selected location"""
        try:
            return self.location_data[country][state][district]
        except KeyError:
            return [(28.6139, 77.2090)]  # Default to Delhi
    
    def generate_traffic_for_location(self, coordinates, num_points=None):
        """Generate traffic data for specific location"""
        data = []
        base_lat, base_lon = coordinates[0]
        
        # Use current time for dynamic seed to simulate real updates
        current_minute = datetime.now().minute
        current_hour = datetime.now().hour
        np.random.seed(hash(f"{base_lat}_{base_lon}_{current_minute}") % 1000)
        
        # Dynamic number of data points based on time and traffic
        if num_points is None:
            base_points = 25
            # More data points during peak hours
            if current_hour in [7, 8, 9, 17, 18, 19]:
                base_points += 15
            # Add random variation
            num_points = base_points + np.random.randint(-5, 10)
            num_points = max(15, min(50, num_points))  # Keep between 15-50
        
        for i in range(num_points):
            lat = base_lat + np.random.normal(0, 0.01)
            lon = base_lon + np.random.normal(0, 0.01)
            
            hour = datetime.now().hour
            is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            is_weekend = 1 if datetime.now().weekday() >= 5 else 0
            
            # Dynamic congestion based on time
            base_congestion = np.random.randint(0, 5)
            if is_peak and not is_weekend:
                base_congestion = min(4, base_congestion + 1)
            elif is_weekend:
                base_congestion = max(0, base_congestion - 1)
            
            # Add time-based variation
            time_factor = np.sin(2 * np.pi * current_minute / 60) * 0.5
            congestion = max(0, min(4, int(base_congestion + time_factor)))
            
            speed = max(5, 60 - (congestion * 12) + np.random.normal(0, 5))
            
            data.append({
                'lat': lat,
                'lon': lon,
                'congestion_level': congestion,
                'average_speed': speed,
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                'hour': hour,
                'is_peak_hour': is_peak,
                'is_weekend': is_weekend,
                'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog'], p=[0.7, 0.2, 0.1])
            })
        
        return pd.DataFrame(data)
    
    def load_data(self):
        """Load or generate traffic data"""
        if 'traffic_data' not in st.session_state:
            # Generate sample data
            st.session_state.traffic_data = self.collector.generate_sample_data(1000)
        return st.session_state.traffic_data
    
    def create_traffic_map(self, df):
        """Create interactive traffic map"""
        # Create base map
        center_lat = df['lat'].mean()
        center_lon = df['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Color mapping for congestion levels
        colors = {0: 'green', 1: 'lightgreen', 2: 'yellow', 3: 'orange', 4: 'red'}
        
        # Add markers for each traffic point
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                popup=f"Congestion: {CONGESTION_LEVELS[row['congestion_level']]}<br>"
                      f"Speed: {row['average_speed']:.1f} km/h<br>"
                      f"Time: {row['timestamp']}",
                color=colors[row['congestion_level']],
                fill=True,
                fillColor=colors[row['congestion_level']],
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Traffic Levels</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Free Flow</p>
        <p><i class="fa fa-circle" style="color:lightgreen"></i> Light Traffic</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Moderate Traffic</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Heavy Traffic</p>
        <p><i class="fa fa-circle" style="color:red"></i> Severe Congestion</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_time_series_plot(self, df):
        """Create time series plot of traffic patterns"""
        # Extract hour from timestamp if hour column doesn't exist
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        
        # Group by hour and calculate average congestion
        hourly_traffic = df.groupby('hour')['congestion_level'].mean().reset_index()
        
        fig = px.line(
            hourly_traffic, 
            x='hour', 
            y='congestion_level',
            title='Average Traffic Congestion by Hour',
            labels={'hour': 'Hour of Day', 'congestion_level': 'Average Congestion Level'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            yaxis=dict(range=[0, 4])
        )
        
        return fig
    
    def create_congestion_distribution(self, df):
        """Create congestion level distribution chart"""
        congestion_counts = df['congestion_level'].value_counts().sort_index()
        congestion_labels = [CONGESTION_LEVELS[i] for i in congestion_counts.index]
        
        fig = px.pie(
            values=congestion_counts.values,
            names=congestion_labels,
            title='Traffic Congestion Distribution',
            color_discrete_sequence=['green', 'lightgreen', 'yellow', 'orange', 'red']
        )
        
        return fig
    
    def create_weather_impact_plot(self, df):
        """Create weather impact visualization"""
        if 'weather_condition' in df.columns and len(df['weather_condition'].unique()) > 1:
            weather_impact = df.groupby('weather_condition')['congestion_level'].mean().reset_index()
            
            fig = px.bar(
                weather_impact,
                x='weather_condition',
                y='congestion_level',
                title='Average Congestion Level by Weather Condition',
                color='congestion_level',
                color_continuous_scale='RdYlGn_r'
            )
        else:
            # Create a simple bar chart if weather data is not available
            fig = px.bar(
                x=['Clear', 'Rain', 'Fog'], 
                y=[2.1, 2.8, 3.2],
                title='Average Congestion Level by Weather Condition',
                labels={'x': 'Weather Condition', 'y': 'Congestion Level'}
            )
        
        return fig
    
    def predict_traffic(self, lat, lon, hour, day_of_week, temperature, weather_condition):
        """Predict traffic congestion for given parameters"""
        # Create input data
        input_data = pd.DataFrame({
            'lat': [lat],
            'lon': [lon],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [1 if day_of_week >= 5 else 0],
            'is_peak_hour': [1 if hour in [7, 8, 9, 17, 18, 19] else 0],
            'temperature': [temperature],
            'humidity': [60],  # Default value
            'wind_speed': [10],  # Default value
            'hour_sin': [np.sin(2 * np.pi * hour / 24)],
            'hour_cos': [np.cos(2 * np.pi * hour / 24)],
            'day_sin': [np.sin(2 * np.pi * day_of_week / 7)],
            'day_cos': [np.cos(2 * np.pi * day_of_week / 7)],
            'weather_impact': [1.0 if weather_condition == 'Clear' else 0.8],
            'is_rush_hour': [1 if hour in [7, 8, 9, 17, 18, 19] else 0],
            'weather_condition_encoded': [0 if weather_condition == 'Clear' else 1]
        })
        
        try:
            # Normalize input data
            input_normalized = self.preprocessor.scaler.transform(input_data)
            input_df = pd.DataFrame(input_normalized, columns=input_data.columns)
            
            # Make prediction
            prediction = self.models.predict(input_df)
            return prediction[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 2  # Default to moderate traffic
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.set_page_config(
            page_title="Real-Time Traffic Prediction System",
            page_icon="🚗",
            layout="wide"
        )
        
        st.title("🚗 Real-Time Traffic Prediction System")
        st.markdown("---")
        
        # Sidebar for controls
        st.sidebar.header("Location Search")
        
        # Location selection
        countries = list(self.location_data.keys())
        selected_country = st.sidebar.selectbox("Select Country", countries)
        
        states = list(self.location_data[selected_country].keys())
        selected_state = st.sidebar.selectbox("Select State/Region", states)
        
        districts = list(self.location_data[selected_country][selected_state].keys())
        selected_district = st.sidebar.selectbox("Select District/City", districts)
        
        # Get coordinates for selected location
        location_coords = self.get_coordinates_for_location(selected_country, selected_state, selected_district)
        
        st.sidebar.header("Controls")
        
        # Auto refresh settings (define before use)
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.selectbox("Refresh Interval", [30, 60, 120], index=1)
        
        # Generate traffic data with caching
        cache_key = f'traffic_data_{selected_country}_{selected_state}_{selected_district}'
        if cache_key not in st.session_state:
            st.session_state[cache_key] = self.generate_traffic_for_location(location_coords)
        df = st.session_state[cache_key]
        
        # Display selected location info with live status
        current_time = datetime.now().strftime("%H:%M:%S")
        data_points = len(df)
        st.sidebar.success(f"Selected: {selected_district}, {selected_state}, {selected_country}")
        st.sidebar.metric("Live Data Points", data_points)
        if auto_refresh:
            st.sidebar.info(f"🔄 Live Updates ON | {current_time}")
        else:
            st.sidebar.info(f"⏸️ Live Updates OFF | {current_time}")
        
        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            # Force new data generation
            st.session_state[cache_key] = self.generate_traffic_for_location(location_coords)
            st.success("Data refreshed!")
            st.rerun()
        
        # Auto refresh logic without screen blur
        if auto_refresh:
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            current_time_sec = time.time()
            if current_time_sec - st.session_state.last_refresh > refresh_interval:
                # Force regenerate with new data points
                st.session_state[cache_key] = self.generate_traffic_for_location(location_coords)
                st.session_state.last_refresh = current_time_sec
                st.rerun()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Live Traffic Map - {selected_district}, {selected_state}")
            
            # Create and display map with smooth updates
            map_container = st.container()
            with map_container:
                traffic_map = self.create_traffic_map(df)
                map_data = st_folium(
                    traffic_map, 
                    width=700, 
                    height=500,
                    returned_objects=[],
                    key=f"traffic_map_{selected_country}_{selected_state}_{selected_district}"
                )
            
            # Location statistics
            st.subheader("Location Statistics")
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                avg_congestion = df['congestion_level'].mean()
                st.metric("Avg Congestion", f"{avg_congestion:.1f}/4")
            
            with col1b:
                avg_speed = df['average_speed'].mean()
                st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
            
            with col1c:
                total_points = len(df)
                # Show dynamic status
                current_hour = datetime.now().hour
                status = "Peak" if current_hour in [7, 8, 9, 17, 18, 19] else "Normal"
                st.metric("Data Points", f"{total_points} ({status})")
        
        with col2:
            st.subheader("Traffic Prediction")
            
            # Prediction inputs
            lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
            hour = st.slider("Hour", 0, 23, datetime.now().hour)
            day_of_week = st.selectbox("Day of Week", 
                                     options=list(range(7)),
                                     format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
            temperature = st.slider("Temperature (°C)", -10, 40, 20)
            weather = st.selectbox("Weather", ['Clear', 'Rain', 'Snow', 'Fog'])
            
            if st.button("Predict Traffic"):
                prediction = self.predict_traffic(lat, lon, hour, day_of_week, temperature, weather)
                
                st.success(f"Predicted Congestion Level: **{CONGESTION_LEVELS[prediction]}**")
                
                # Show prediction confidence
                confidence_colors = {0: 'green', 1: 'lightgreen', 2: 'yellow', 3: 'orange', 4: 'red'}
                st.markdown(f"<div style='padding: 10px; background-color: {confidence_colors[prediction]}; "
                          f"border-radius: 5px; text-align: center; color: black; font-weight: bold;'>"
                          f"{CONGESTION_LEVELS[prediction]}</div>", unsafe_allow_html=True)
            
            # Quick location predictions
            st.subheader("Quick Predictions")
            if st.button(f"Current Traffic in {selected_district}"):
                current_hour = datetime.now().hour
                current_day = datetime.now().weekday()
                
                # Use first coordinate of selected location
                coord = location_coords[0]
                pred = self.predict_traffic(coord[0], coord[1], current_hour, current_day, 25, 'Clear')
                
                st.info(f"Current traffic in {selected_district}: **{CONGESTION_LEVELS[pred]}**")
            
            # Hourly prediction for selected location
            st.subheader("24-Hour Forecast")
            if st.button("Generate 24h Forecast"):
                forecast_data = []
                coord = location_coords[0]
                current_day = datetime.now().weekday()
                
                for hour in range(24):
                    pred = self.predict_traffic(coord[0], coord[1], hour, current_day, 25, 'Clear')
                    forecast_data.append({
                        'Hour': f"{hour:02d}:00",
                        'Predicted Congestion': CONGESTION_LEVELS[pred],
                        'Level': pred
                    })
                
                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df, use_container_width=True)
                
                # Plot hourly forecast
                import plotly.express as px
                fig = px.line(forecast_df, x='Hour', y='Level', 
                            title=f'24-Hour Traffic Forecast - {selected_district}',
                            labels={'Level': 'Congestion Level (0-4)'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Analytics section
        st.markdown("---")
        st.subheader("Traffic Analytics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Time series plot
            time_series_fig = self.create_time_series_plot(df)
            st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Weather impact plot
            weather_fig = self.create_weather_impact_plot(df)
            st.plotly_chart(weather_fig, use_container_width=True)
        
        with col4:
            # Congestion distribution
            dist_fig = self.create_congestion_distribution(df)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Key metrics
            st.subheader("Key Metrics")
            avg_congestion = df['congestion_level'].mean()
            
            # Safe calculation for peak hour and weekend congestion
            if 'is_peak_hour' in df.columns:
                peak_data = df[df['is_peak_hour'] == 1]
                peak_hour_congestion = peak_data['congestion_level'].mean() if len(peak_data) > 0 else avg_congestion
            else:
                peak_hour_congestion = avg_congestion
            
            if 'is_weekend' in df.columns:
                weekend_data = df[df['is_weekend'] == 1]
                weekend_congestion = weekend_data['congestion_level'].mean() if len(weekend_data) > 0 else avg_congestion
            else:
                weekend_congestion = avg_congestion
            
            st.metric("Average Congestion Level", f"{avg_congestion:.2f}")
            st.metric("Peak Hour Congestion", f"{peak_hour_congestion:.2f}")
            st.metric("Weekend Congestion", f"{weekend_congestion:.2f}")
        
        # Traffic Analytics Table
        st.markdown("---")
        st.subheader("Traffic Analytics")
        
        # Create analytics data
        analytics_data = []
        for _, row in df.iterrows():
            analytics_data.append({
                'Time': row['timestamp'].strftime('%H:%M'),
                'Location': f"{selected_district}, {selected_state}",
                'Latitude': f"{row['lat']:.4f}",
                'Longitude': f"{row['lon']:.4f}",
                'Speed (km/h)': f"{row['average_speed']:.1f}",
                'Congestion Level': CONGESTION_LEVELS[row['congestion_level']],
                'Traffic Status': 'Heavy' if row['congestion_level'] >= 3 else 'Moderate' if row['congestion_level'] >= 2 else 'Light'
            })
        
        analytics_df = pd.DataFrame(analytics_data)
        
        # Display analytics table with filters
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            status_filter = st.selectbox("Filter by Traffic Status", 
                                       ['All', 'Light', 'Moderate', 'Heavy'])
        
        with col_filter2:
            congestion_filter = st.selectbox("Filter by Congestion Level", 
                                           ['All'] + list(CONGESTION_LEVELS.values()))
        
        # Apply filters
        filtered_df = analytics_df.copy()
        if status_filter != 'All':
            filtered_df = filtered_df[filtered_df['Traffic Status'] == status_filter]
        if congestion_filter != 'All':
            filtered_df = filtered_df[filtered_df['Congestion Level'] == congestion_filter]
        
        st.write(f"Showing {len(filtered_df)} of {len(analytics_df)} records")
        st.dataframe(filtered_df.head(15), use_container_width=True)
        
        # Traffic summary statistics
        st.subheader("Traffic Summary")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            free_flow = len(df[df['congestion_level'] == 0])
            st.metric("Free Flow Areas", free_flow)
        
        with col_sum2:
            heavy_traffic = len(df[df['congestion_level'] >= 3])
            st.metric("Heavy Traffic Areas", heavy_traffic)
        
        with col_sum3:
            avg_speed_all = df['average_speed'].mean()
            st.metric("Average Speed", f"{avg_speed_all:.1f} km/h")
        
        with col_sum4:
            peak_congestion = df['congestion_level'].max()
            st.metric("Peak Congestion", CONGESTION_LEVELS[peak_congestion])
        
        # Location comparison
        st.markdown("---")
        st.subheader("Compare Locations")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            comp_country = st.selectbox("Compare Country", countries, key="comp_country")
            comp_states = list(self.location_data[comp_country].keys())
            comp_state = st.selectbox("Compare State", comp_states, key="comp_state")
            comp_districts = list(self.location_data[comp_country][comp_state].keys())
            comp_district = st.selectbox("Compare District", comp_districts, key="comp_district")
        
        with col_comp2:
            if st.button("Compare Traffic"):
                comp_coords = self.get_coordinates_for_location(comp_country, comp_state, comp_district)
                comp_df = self.generate_traffic_for_location(comp_coords)
                
                # Comparison metrics
                st.write(f"**{selected_district}**: Avg Congestion {df['congestion_level'].mean():.1f}")
                st.write(f"**{comp_district}**: Avg Congestion {comp_df['congestion_level'].mean():.1f}")
                
                if df['congestion_level'].mean() > comp_df['congestion_level'].mean():
                    st.warning(f"{selected_district} has higher traffic than {comp_district}")
                else:
                    st.success(f"{comp_district} has higher traffic than {selected_district}")

if __name__ == "__main__":
    dashboard = TrafficDashboard()
    dashboard.run_dashboard()