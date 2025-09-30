Milestone 1: Week 1-2
Module 1: Data Collection from APIs and location databases
• Collect air quality data (PM2.5, PM10, NO₂, CO, SO₂, O₃) from the OpenAQ API for selected locations. •
Collect weather data (temperature, humidity, wind speed, wind direction) from the OpenWeatherMap
API.
• Extract nearby physical features such as roads, industrial zones, dump sites, and agricultural fields using
OpenStreetMap via OSMnx.
• Tag each data point with latitude, longitude, timestamp, and source API metadata. • Store
the collected data in structured CSV/JSON format for preprocessing and modeling.
solution:
   Milestone 1: Virtual Dataset Creation

To predict pollution sources, we created a virtual dataset by combining multiple data sources:

Air Pollution Data – PM2.5, PM10, NO2, CO, SO2, O3 from OpenWeatherMap Air Pollution API.

Weather Data – Temperature, Humidity, Wind Speed & Direction, Cloud Cover from OpenWeatherMap Weather API.

Geospatial Features – Counts of roads, industrial zones, dump sites, and farmland using OSMnx (OpenStreetMap).

City Coordinates – Latitude and longitude from a Kaggle India cities dataset.

Workflow:

For each city, we pass lat/lon to the APIs and city name to OSMnx.

Collect all features into a single row with timestamp.

Save as CSV (city_air_weather_osm.csv) for downstream feature engineering, labeling, and ML model training.

This dataset provides structured, multi-source input to enable pollution source prediction and geospatial analysis.
                  
