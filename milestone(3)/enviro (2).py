

import pandas as pd
import folium
from folium.plugins import HeatMap
import os

# --- Step 1 & 2: Load and Prepare Data ---
# We'll load the labeled dataset which contains location and pollution data,
# and the 'pollution_source' label.
try:
    df = pd.read_csv("/content/city_air_weather_osm_labeled.csv")
    print("Labeled data loaded successfully for mapping.")
    # Ensure necessary columns are present
    if not all(col in df.columns for col in ['lat', 'lon', 'pm2_5', 'pollution_source']):
        print("Error: Required columns (lat, lon, pm2_5, pollution_source) not found in the dataset.")
        df = None # Set df to None to prevent further errors
except FileNotFoundError:
    print("Error: The file '/content/city_air_weather_osm_labeled.csv' was not found.")
    print("Please make sure the labeling step was completed successfully and the file exists.")
    df = None


if df is not None:
    # --- Step 3: Initialize Map ---
    # Find the center of India for map initialization
    # Using the mean latitude and longitude from the dataset
    map_center_lat = df['lat'].mean()
    map_center_lon = df['lon'].mean()

    # Create a base map using Folium
    india_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=5)
    print("Map initialized.")

    # --- Step 4: Create Pollution Heatmap (using PM2.5 as an example) ---
    print("Adding Heatmap layer...")
    # Prepare data for heatmap: list of [latitude, longitude, value]
    # Using PM2.5 for the heatmap intensity
    heatmap_data = df[['lat', 'lon', 'pm2_5']].values.tolist()

    # Add HeatMap layer to the map
    HeatMap(heatmap_data).add_to(india_map)
    print("Heatmap layer added.")


    # --- Step 5: Add Source-Specific Markers ---
    print("Adding source-specific markers...")

    # Define colors or icons for different pollution sources
    source_colors = {
        'Vehicular': 'blue',
        'Industrial': 'red',
        'Agricultural': 'green',
        'Other': 'gray' # Default color
    }

    # Add a layer for markers so they can be toggled on/off
    marker_layer = folium.FeatureGroup(name='Pollution Sources').add_to(india_map)

    # Iterate through each city and add a marker
    for index, row in df.iterrows():
        city_name = row['city']
        latitude = row['lat']
        longitude = row['lon']
        pollution_source = row['pollution_source']
        pm2_5_value = row['pm2_5']

        # Get the color based on the pollution source
        marker_color = source_colors.get(pollution_source, 'gray') # Use gray for unknown sources

        # Create a popup with information
        popup_text = f"<b>City:</b> {city_name}<br>" \
                     f"<b>Predicted Source:</b> {pollution_source}<br>" \
                     f"<b>PM2.5:</b> {pm2_5_value:.2f}" # Format PM2.5 value


        # Add a marker to the marker layer
        folium.Marker(
            location=[latitude, longitude],
            popup=popup_text,
            icon=folium.Icon(color=marker_color, icon='info-sign') # Using a simple icon with color
        ).add_to(marker_layer)

    print("Source-specific markers added.")


    # --- Step 6: Visualize High-Risk Zones (Basic Implementation) ---
    # A simple way to visualize higher risk areas is to use different marker sizes or colors
    # based on pollutant severity. We already used color for source, let's use size (radius)
    # for PM2.5 severity on a separate layer.

    print("Adding High PM2.5 zones visualization...")

    high_pm_layer = folium.FeatureGroup(name='High PM2.5 Zones').add_to(india_map)

    # Define a simple rule for "high PM2.5" - e.g., above the 75th percentile of the scaled data
    pm2_5_threshold = df['pm2_5'].quantile(0.75)

    for index, row in df.iterrows():
        latitude = row['lat']
        longitude = row['lon']
        pm2_5_value = row['pm2_5']

        # If PM2.5 is above the threshold, add a larger circle marker
        if pm2_5_value > pm2_5_threshold:
             folium.CircleMarker(
                location=[latitude, longitude],
                radius=10, # Larger radius for high PM2.5
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.6,
                popup=f"<b>High PM2.5:</b> {pm2_5_value:.2f}"
            ).add_to(high_pm_layer)

    print("High PM2.5 zones visualization added.")


    # --- Step 7: Add Interactive Controls (Basic Layer Control) ---
    # Add a layer control to toggle layers
    folium.LayerControl().add_to(india_map)
    print("Layer control added.")


    # --- Step 8: Export Map ---
    output_map_file = "pollution_map.html"
    india_map.save(output_map_file)
    print(f"\nInteractive map saved to '{output_map_file}'")
    print(f"You can view the map by downloading '{output_map_file}' from the Files tab and opening it in a web browser.")

    # --- Step 9: Finish Task (Implicit in the output file) ---
    print("\nModule 5 tasks completed in this code block.")
    print("The generated map file visualizes the pollution data and predicted sources.")

# Set your OpenWeatherMap API key in Colab
os.environ["OWM_API_KEY"] = "13bd8cad6b6a28245eff4047fb57163f"

!pip install streamlit

Commented out IPython magic to ensure Python compatibility.
  %%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import os
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Air Pollution Dashboard", layout="wide")

st.title("Indian Cities Air Pollution Dashboard")
st.write("Explore real-time air quality, predicted pollution sources, and historical insights.")

# -----------------------------
# Load model
# -----------------------------
model_path = "best_pollution_source_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.warning(f"⚠️ Model file '{model_path}' not found. Prediction will not work.")
    model = None

# -----------------------------
# Load historical data
# -----------------------------
data_path = "/content/city_air_weather_osm_labeled.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.success("✅ Historical data loaded!")
else:
    st.warning(f"⚠️ Data file '{data_path}' not found. Some features will use default values.")
    df = None

# -----------------------------
# Scaler
# -----------------------------
scaler_cols = ['pm2_5','pm10','no2','co','so2','o3',
               'temp_day','temp_min','temp_max','humidity','pressure',
               'wind_speed','wind_deg','clouds',
               'road_density','industrial_density','dump_density','farmland_density']

if df is not None:
    scaler = MinMaxScaler()
    try:
        features_for_scaler = df.drop(['city','lat','lon','timestamp','city_code','pollution_source'], axis=1)
        features_for_scaler = features_for_scaler.fillna(features_for_scaler.median())
        scaler.fit(features_for_scaler[scaler_cols])
    except:
        st.warning("Could not fit scaler, predictions may fail.")
        scaler = None
else:
    scaler = None

# -----------------------------
# OpenWeatherMap API Key
# -----------------------------
OWM_KEY = "13bd8cad6b6a28245eff4047fb57163f"
OWM_KEY = st.secrets.get("OWM_API_KEY", None)
if not OWM_KEY:
    st.warning("⚠️ OpenWeatherMap API key not found. Real-time data unavailable.")

# -----------------------------
# API Functions
# -----------------------------
def get_air_quality(lat, lon):
    if not OWM_KEY: return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_KEY}"
        return requests.get(url).json()
    except:
        return None

def get_weather(lat, lon):
    if not OWM_KEY: return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
        return requests.get(url).json()
    except:
        return None

# -----------------------------
# OSM Feature Helper
# -----------------------------
def get_osm(city_name):
    if df is not None:
        city_data = df[df['city'].str.lower() == city_name.lower()]
        if not city_data.empty:
            return city_data[['road_density','industrial_density','dump_density','farmland_density']].iloc[0].to_dict()
    return {'road_density':0,'industrial_density':0,'dump_density':0,'farmland_density':0}

# -----------------------------
# User input
# -----------------------------
st.header("Enter City or Coordinates")
input_type = st.radio("Input method:", ["City Name", "Latitude & Longitude"])

lat = lon = None
city_name = None

if input_type == "City Name":
    city_name = st.text_input("City Name (e.g., Delhi)")
    if city_name and df is not None:
        coords = df[df['city'].str.lower()==city_name.lower()][['lat','lon']]
        if not coords.empty:
            lat, lon = coords.iloc[0]
            st.info(f"Coordinates: {lat}, {lon}")
else:
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Get Pollution Data & Predict") and lat is not None and lon is not None:

    st.subheader("Real-time Data & Prediction")

    air_data = get_air_quality(lat, lon)
    weather_data = get_weather(lat, lon)

    if air_data and weather_data and model and scaler:
        try:
            p = air_data['list'][0]['components']
            w = weather_data['main']
            wind = weather_data['wind']
            clouds = weather_data['clouds']

            osm = get_osm(city_name if city_name else "Unknown")

            feat = {
                'pm2_5': p.get('pm2_5',0),
                'pm10': p.get('pm10',0),
                'no2': p.get('no2',0),
                'co': p.get('co',0),
                'so2': p.get('so2',0),
                'o3': p.get('o3',0),
                'temp_day': w.get('temp',0),
                'temp_min': w.get('temp_min',w.get('temp',0)),
                'temp_max': w.get('temp_max',w.get('temp',0)),
                'humidity': w.get('humidity',0),
                'pressure': w.get('pressure',0),
                'wind_speed': wind.get('speed',0),
                'wind_deg': wind.get('deg',0),
                'clouds': clouds.get('all',0),
                'road_density': osm.get('road_density',0),
                'industrial_density': osm.get('industrial_density',0),
                'dump_density': osm.get('dump_density',0),
                'farmland_density': osm.get('farmland_density',0)
            }

            feat_df = pd.DataFrame([feat])
            feat_scaled = scaler.transform(feat_df[scaler_cols])
            pred_enc = model.predict(feat_scaled)[0]
            pred_proba = model.predict_proba(feat_scaled)[0]
            pred_label = model.classes_[pred_enc]

            st.success(f"Predicted Pollution Source: {pred_label}")
            st.write("Confidence Scores:")
            st.dataframe(pd.DataFrame({'Source': model.classes_, 'Confidence': pred_proba}))

            st.subheader("Real-time Pollutant Levels")
            st.json({k:v for k,v in p.items() if k in ['pm2_5','pm10','no2','co','so2','o3']})

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Data/model unavailable. Check API key, model file, and data file.")

# -----------------------------
# Placeholders for charts/map
# -----------------------------
st.header("Pollutant Trends Over Time (Coming Soon)")
st.line_chart(pd.DataFrame({'pm2_5':[0,1,2],'pm10':[0,1,2]}))

st.header("Predicted Source Distribution (Coming Soon)")
st.bar_chart(pd.DataFrame({'Industrial':[1,2,3],'Vehicular':[3,2,1]}))

st.header("Geospatial Map (Coming Soon)")
st.map(pd.DataFrame({'lat':[lat if lat else 28],'lon':[lon if lon else 77]}))

Commented out IPython magic to ensure Python compatibility.
%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import os
import requests
from datetime import datetime, timezone
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import base64 # To encode file for download

# Import modules for sending emails
import smtplib
from email.mime.text import MIMEText


# -----------------------------
# Load the trained model
# -----------------------------
model_filename = 'best_pollution_source_model.joblib'
model = None
if os.path.exists(model_filename):
    try:
        model = joblib.load(model_filename)
        # st.success("Trained model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model file '{model_filename}': {e}")
else:
    st.error(f"Error: Model file '{model_filename}' not found.")


# -----------------------------
# Load historical data for scaler fitting and historical views
# -----------------------------
data_filename = '/content/city_air_weather_osm_labeled.csv' # Use the labeled data for historical views and scaler fitting
df_historical = None
scaler = None
if os.path.exists(data_filename):
    try:
        df_historical = pd.read_csv(data_filename)
        # st.success("Historical data loaded successfully!")

        # Fit scaler on historical data
        scaler_cols = ['pm2_5','pm10','no2','co','so2','o3',
                       'temp_day','temp_min','temp_max','humidity','pressure',
                       'wind_speed','wind_deg','clouds',
                       'road_density','industrial_density','dump_density','farmland_density']

        # Ensure columns exist before selecting
        cols_to_scale = [col for col in scaler_cols if col in df_historical.columns]
        if cols_to_scale:
            scaler = MinMaxScaler()
            features_for_scaler_fit = df_historical[cols_to_scale].fillna(df_historical[cols_to_scale].median())
            scaler.fit(features_for_scaler_fit)
            # st.success("Scaler fitted successfully on historical data!")
        else:
            st.warning("No valid columns found for scaler fitting.")
            scaler = None

    except Exception as e:
        st.warning(f"Could not load and fit scaler from historical data: {e}")
        df_historical = None
        scaler = None
else:
    st.warning(f"Warning: Historical data file '{data_filename}' not found. Some features may be limited.")
    df_historical = None
    scaler = None


# -----------------------------
# OpenWeatherMap API Key Handling
# -----------------------------
# Read OWM_KEY from environment variable set in the Colab notebook
OWM_KEY = os.environ.get("OWM_API_KEY") # Read from environment variable

if not OWM_KEY:
    st.warning("OpenWeatherMap API key environment variable not set.")
    # Add a text input for the user to enter the API key in the dashboard
    OWM_KEY = st.text_input("Enter your OpenWeatherMap API Key:")
    if OWM_KEY:
        # If the user enters the key, set it as an environment variable for the current session
        os.environ["OWM_API_KEY"] = OWM_KEY
        st.success("OpenWeatherMap API Key set from input!")
    else:
        st.info("Please enter your OpenWeatherMap API Key to fetch real-time data.")

# -----------------------------
# Email Credentials Handling
# -----------------------------
# Read email credentials from environment variables set in the Colab notebook secrets
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587)) # Default SMTP port
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

email_configured = all([SMTP_SERVER, EMAIL_ADDRESS, EMAIL_PASSWORD])

if not email_configured:
    st.warning("Email credentials environment variables not fully set.")
    st.info("Please set SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, and EMAIL_PASSWORD in your Colab secrets (`🔑 > Add a secret`) for email alerts.")


# -----------------------------
# API functions
# -----------------------------
def get_air_quality(lat, lon, api_key):
    if not api_key: return None
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching air quality data: {e}")
        return None

def get_weather(lat, lon, api_key):
    if not api_key: return None
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# -----------------------------
# Get OSM features from historical data (using preprocessed data)
# -----------------------------
def get_osm_features_from_data(city_name, df_historical):
    if df_historical is not None:
        city_data = df_historical[df_historical['city'].str.lower() == city_name.lower()]
        if not city_data.empty:
            # Ensure columns exist before selecting
            osm_cols = ['road_density', 'industrial_density', 'dump_density', 'farmland_density']
            present_osm_cols = [col for col in osm_cols if col in city_data.columns]
            if present_osm_cols:
                 return city_data[present_osm_cols].iloc[0].to_dict()
    return {
        'road_density': None,
        'industrial_density': None,
        'dump_density': None,
        'farmland_density': None
    }

# -----------------------------
# Function to send email alert
# -----------------------------
def send_email_alert(to_email, subject, body, smtp_server, smtp_port, from_email, password):
    if not all([to_email, subject, body, smtp_server, from_email, password]):
        st.warning("Email credentials or recipient not fully configured. Cannot send email.")
        return

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls() # Secure the connection
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        st.success(f"Email alert sent to {to_email}")
    except Exception as e:
        st.error(f"Error sending email alert: {e}")


# -----------------------------
# Function to generate a sample report (CSV)
# -----------------------------
def generate_report(data):
    # This is a placeholder. In a real application, you would format
    # the data and include relevant insights or summaries.
    if data is not None:
        return data.to_csv(index=False).encode('utf-8')
    return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide") # Use wide layout
st.title("Indian Cities Air Pollution Dashboard")
st.write("Explore real-time air quality data, predicted pollution sources, and historical insights.")

st.header("Enter City or Coordinates")
input_option = st.radio("Choose input method:", ("City Name", "Latitude and Longitude"))

city_name = None
lat = None
lon = None

if input_option == "City Name":
    city_name = st.text_input("Enter City Name (e.g., Delhi, Mumbai)")
    if city_name and df_historical is not None:
        city_coords = df_historical[df_historical['city'].str.lower() == city_name.lower()][['lat', 'lon']]
        if not city_coords.empty:
            lat = city_coords.iloc[0]['lat']
            lon = city_coords.iloc[0]['lon']
            st.write(f"Coordinates: Latitude {lat:.4f}, Longitude {lon:.4f}")
        else:
            st.warning("City not found in historical data. Try Lat/Lon input.")
            city_name = None
    elif city_name and df_historical is None:
        st.warning("Historical data not loaded. Cannot resolve city name.")
        city_name = None
else:
    lat = st.number_input("Enter Latitude", format="%.6f")
    lon = st.number_input("Enter Longitude", format="%.6f")

# -----------------------------
# Fetch data and predict
# -----------------------------
if (lat is not None and lon is not None):
    if st.button("Get Pollution Data and Predict Source"):
        st.header("Real-time Data and Prediction")

        # Check if API key is available before fetching data
        if OWM_KEY:
            air_data = get_air_quality(lat, lon, OWM_KEY)
            weather_data = get_weather(lat, lon, OWM_KEY)

            if air_data and weather_data and model and scaler and df_historical is not None:
                try:
                    p = air_data['list'][0]['components']
                    w_main = weather_data['main']
                    w_wind = weather_data['wind']
                    w_clouds = weather_data['clouds']

                    osm_features = get_osm_features_from_data(city_name if city_name else "Unknown City", df_historical)

                    feature_vector_data = {
                        'pm2_5': p.get('pm2_5'),
                        'pm10': p.get('pm10'),
                        'no2': p.get('no2'),
                        'co': p.get('co'),
                        'so2': p.get('so2'),
                        'o3': p.get('o3'),
                        'temp_day': w_main.get('temp'),
                        'temp_min': w_main.get('temp_min',w_main.get('temp',0)),
                        'temp_max': w_main.get('temp_max',w_main.get('temp',0)),
                        'humidity': w_main.get('humidity'),
                        'pressure': w_main.get('pressure'),
                        'wind_speed': w_wind.get('speed'),
                        'wind_deg': w_wind.get('deg', 0),
                        'clouds': w_clouds.get('all'),
                        'road_density': osm_features.get('road_density', df_historical['road_density'].median() if df_historical is not None and 'road_density' in df_historical.columns else None),
                        'industrial_density': osm_features.get('industrial_density', df_historical['industrial_density'].median() if df_historical is not None and 'industrial_density' in df_historical.columns else None),
                        'dump_density': osm_features.get('dump_density', df_historical['dump_density'].median() if df_historical is not None and 'dump_density' in df_historical.columns else None),
                        'farmland_density': osm_features.get('farmland_density', df_historical['farmland_density'].median() if df_historical is not None and 'farmland_density' in df_historical.columns else None)
                    }

                    feature_df = pd.DataFrame([feature_vector_data])

                    # Ensure columns expected by scaler exist and handle NaNs before scaling
                    cols_to_scale = [col for col in scaler_cols if col in feature_df.columns]
                    if cols_to_scale:
                         feature_df[cols_to_scale] = feature_df[cols_to_scale].fillna(df_historical[cols_to_scale].median() if df_historical is not None and all(col in df_historical.columns for col in cols_to_scale) else 0)
                         scaled_features = scaler.transform(feature_df[cols_to_scale])
                         scaled_features_df = pd.DataFrame(scaled_features, columns=cols_to_scale)

                         # Make prediction
                         prediction_label_encoded = model.predict(scaled_features_df)[0]
                         prediction_proba = model.predict_proba(scaled_features_df)[0]

                         # Decode the predicted label
                         # Need access to target_classes from training - assuming 'Other', 'Industrial', 'Vehicular', 'Agricultural' order
                         # In a real app, save and load target_classes or use a LabelEncoder
                         # For this example, we'll hardcode based on previous output
                         target_classes = ['Other', 'Industrial', 'Vehicular', 'Agricultural']
                         prediction_label = target_classes[prediction_label_encoded]


                         st.subheader("Prediction Results")
                         st.write(f"Predicted Pollution Source: **{prediction_label}**")

                         st.write("Confidence Scores:")
                         confidence_df = pd.DataFrame({'Source': target_classes, 'Confidence': prediction_proba})
                         st.dataframe(confidence_df.sort_values(by='Confidence', ascending=False))

                         st.subheader("Real-time Pollutant Levels")
                         current_pollutants = {k: v for k, v in p.items() if k in ['pm2_5', 'pm10', 'no2', 'co', 'so2', 'o3']}
                         st.json(current_pollutants)

                         st.subheader("Alerts")
                         PM2_5_ALERT_THRESHOLD = 50 # Example threshold for PM2.5
                         pm2_5_level = p.get('pm2_5')
                         if pm2_5_level is not None and pm2_5_level > PM2_5_ALERT_THRESHOLD:
                             st.warning(f"🚨 ALERT: PM2.5 level ({pm2_5_level:.2f}) is high!")

                             # Trigger email alert if configured
                             if email_configured and st.session_state.get('alert_email'):
                                 alert_subject = f"High PM2.5 Alert for {city_name if city_name else 'Specified Location'}"
                                 alert_body = f"High PM2.5 detected at {city_name if city_name else f'Lat: {lat:.2f}, Lon: {lon:.2f}'}: {pm2_5_level:.2f}.\nPredicted source: {prediction_label}.\n\nReal-time data:\n{current_pollutants}"
                                 send_email_alert(st.session_state['alert_email'], alert_subject, alert_body, SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD)
                             elif st.session_state.get('alert_email'):
                                st.info("Email credentials not configured. Email alert not sent.")
                             else:
                                st.info("Enter an email address in the Alerts section to receive email alerts.")

                         else:
                             st.info("✅ PM2.5 within safe threshold.")

                         st.subheader("Weather Conditions")
                         st.write(f"Temperature: {w_main.get('temp', 0):.1f} °C")
                         st.write(f"Humidity: {w_main.get('humidity', 0)}%")
                         st.write(f"Wind Speed: {w_wind.get('speed', 0):.1f} m/s")
                         st.write(f"Clouds: {w_clouds.get('all', 0)}%")

                    else:
                        st.error("Could not prepare features for scaling. Check column names and historical data.")


                except Exception as e:
                    st.error(f"Error during data processing or prediction: {e}")
            else:
                st.warning("Model, scaler, historical data, or OpenWeatherMap API key missing. Cannot make prediction.")
        else:
             st.info("Please enter your OpenWeatherMap API Key above to fetch real-time data and make a prediction.")


# -----------------------------
# Charts and Visualizations
# -----------------------------
st.header("Historical Data Visualizations")

if df_historical is not None:
    st.subheader("Pollutant Trends Over Time (Sample)")
    # This is a placeholder. You would need to aggregate df_historical
    # by time (e.g., daily, weekly) and select specific pollutants.
    # Example: df_daily_avg = df_historical.resample('D', on='timestamp').mean()
    # st.line_chart(df_daily_avg[['pm2_5', 'pm10']])
    # Ensure 'timestamp' is datetime and set as index for plotting
    try:
        df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
        st.line_chart(df_historical[['timestamp', 'pm2_5', 'pm10']].set_index('timestamp'))
    except KeyError:
        st.warning("Timestamp column not found in historical data.")
    except Exception as e:
        st.warning(f"Error plotting historical data: {e}")


    st.subheader("Predicted Source Distribution (Historical)")
    # This uses the pollution_source column from the historical data
    if 'pollution_source' in df_historical.columns:
        source_counts = df_historical['pollution_source'].value_counts()
        st.bar_chart(source_counts) # Using bar chart as Streamlit doesn't have native pie chart
    else:
        st.info("Pollution source labels not available in historical data for distribution chart.")


else:
    st.info("Historical data not loaded. Cannot display historical trends and distributions.")


# -----------------------------
# Geospatial Map
# -----------------------------
st.header("Pollution Map")
map_file = "pollution_map.html"
if os.path.exists(map_file):
    st.write("Interactive map showing pollution data and predicted sources.")
    # Embed the HTML map file
    with open(map_file, "r") as f:
        html_string = f.read()
        st.components.v1.html(html_string, width=800, height=600, scrolling=True)
else:
    st.warning(f"Map file '{map_file}' not found. Please generate the map first.")


# -----------------------------
# Email Alerts Configuration
# -----------------------------
st.header("Set up Alerts")
st.write("Configure email alerts for critical pollution levels.")

# Input for email address
alert_email = st.text_input("Enter Email Address for Alerts (e.g., your_email@example.com)")

# Store email address in session state
if alert_email:
    st.session_state['alert_email'] = alert_email
    st.success(f"Alert email address set to {alert_email}")
else:
    st.session_state['alert_email'] = None
    st.info("Enter a valid email address to enable email alerts.")

if email_configured:
    st.info("Email sending is configured. Email alerts can be sent when critical pollution levels are detected.")
else:
    st.warning("Email sending is not configured. Email alerts will not be sent.")


# -----------------------------
# Download Reports
# -----------------------------
st.header("Download Reports")
if df_historical is not None:
    st.write("Download a CSV report of the historical pollution data.")
    report_data = df_historical # Or process df_historical for a specific report format
    csv_report = generate_report(report_data)

    if csv_report:
        st.download_button(
            label="Download Historical Data CSV",
            data=csv_report,
            file_name='historical_pollution_report.csv',
            mime='text/csv'
        )
    else:
        st.info("Historical data not available to generate report.")
else:
     st.info("Historical data not loaded. Cannot generate reports.")

import os

# Replace "YOUR_OPENWEATHERMAP_API_KEY" with your actual OpenWeatherMap API key
os.environ["OWM_API_KEY"] = "13bd8cad6b6a28245eff4047fb57163f"

print("OpenWeatherMap API key environment variable set.")

!pip install pyngrok --quiet

import threading, time
from pyngrok import ngrok
from google.colab import userdata
import os

# Get the ngrok authtoken from Colab secrets
NGROK_AUTH_TOKEN = userdata.get("NGROK_AUTH_TOKEN")
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("ngrok authtoken set.")
else:
    print("ngrok authtoken not found in Colab secrets.")
    print("Please add your ngrok authtoken to Colab secrets with the name NGROK_AUTH_TOKEN.")
    # You might want to exit or handle this case further if the token is required


# Set OpenWeatherMap API Key Environment Variable
# Read OWM_API_KEY from environment variable or directly set it here
# IMPORTANT: Replace "YOUR_OPENWEATHERMAP_API_KEY" with your actual key
# We will also pass this directly to the streamlit process
OWM_API_KEY = os.environ.get("OWM_API_KEY", "13bd8cad6b6a28245eff4047fb57163f") # Fallback to hardcoded if env var not set
os.environ["OWM_API_KEY"] = OWM_API_KEY # Ensure it's in the current process environment


# Start Streamlit in background thread, passing the OWM_API_KEY environment variable
def run_streamlit():
    # Pass environment variable directly to the shell command
    !OWM_API_KEY=$OWM_API_KEY streamlit run app.py --server.port 8501 --server.headless true

thread = threading.Thread(target=run_streamlit)
thread.start()

time.sleep(10)  # Wait for Streamlit to boot

# Start ngrok tunnel
try:
    public_url = ngrok.connect(8501)
    print("✅ Dashboard live at:", public_url)
except Exception as e:
    print(f"Error starting ngrok tunnel: {e}")

