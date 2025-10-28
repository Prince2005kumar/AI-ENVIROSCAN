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
