import os
import zipfile
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
from plot_utils import plot_aggregated_time_series
from inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_model_from_registry,
)
from config import DATA_DIR
import requests

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"):
    """Downloads and loads the NYC taxi zones shapefile."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"
    
    if not zip_path.exists():
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
    
    if not shapefile_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    
    return gpd.read_file(shapefile_path).to_crs("epsg:4326")

def create_location_dropdown(predictions, geo_df):
    """Creates a dropdown menu to select a taxi zone by name."""
    merged_data = geo_df.merge(predictions, left_on="LocationID", right_on="pickup_location_id", how="left")
    location_options = merged_data[['LocationID', 'zone']].drop_duplicates().dropna().sort_values(by='zone').drop_duplicates().dropna().sort_values(by='zone')
    location_dict = dict(zip(location_options['zone'], location_options['LocationID']))
    
    selected_zone = st.selectbox("Select Taxi Zone:", options=location_dict.keys())
    return location_dict[selected_zone]

def create_taxi_map(shapefile_path, prediction_data):
    """Create an interactive choropleth map of NYC taxi zones with predicted rides."""
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    ).fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)
    
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    ).add_to(m)
    
    folium.GeoJson(
        nyc_zones.to_json(),
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"].get("predicted_demand", 0)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=["zone", "predicted_demand"], aliases=["Zone:", "Predicted Demand:"]),
    ).add_to(m)
    return m

def main():
    st.title("New York Yellow Taxi Cab Demand Prediction")
    
    progress_bar = st.sidebar.progress(0)
    
    with st.spinner("Downloading and loading shape file..."):
        geo_df = load_shape_data_file(DATA_DIR)
        progress_bar.progress(1 / 5)
    
    with st.spinner("Fetching batch of inference data..."):
        features = load_batch_of_features_from_store(pd.Timestamp.now(tz="America/New_York"))
        features["pickup_hour"] = pd.to_datetime(features["pickup_hour"])
        progress_bar.progress(2 / 5)
    
    with st.spinner("Loading model from registry..."):
        model = load_model_from_registry()
        progress_bar.progress(3 / 5)
    
    with st.spinner("Computing model predictions..."):
        predictions = get_model_predictions(model, features)
        progress_bar.progress(4 / 5)
    
    st.sidebar.header("Select Zone for Analysis")
    selected_location_id = create_location_dropdown(predictions, geo_df)
    
    st.subheader("Taxi Ride Predictions Map")
    map_obj = create_taxi_map(DATA_DIR / "taxi_zones" / "taxi_zones.shp", predictions)
    st_folium(map_obj, width=800, height=600, returned_objects=[])
    
    st.subheader("Predicted Demand Over Time")
    if "pickup_location_id" not in predictions.columns:
        predictions["pickup_location_id"] = features["pickup_location_id"]

    predictions["pickup_location_id"] = features["pickup_location_id"]
    fig = plot_aggregated_time_series(features, predictions, row_id=selected_location_id)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Pickup Locations")
    top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
    top10 = top10.merge(geo_df[['LocationID', 'zone']], left_on='pickup_location_id', right_on='LocationID', how='left')
    st.dataframe(top10[['zone', 'pickup_location_id', 'predicted_demand']])
    
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
    with col2:
        st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
    with col3:
        st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")
    
    progress_bar.progress(5 / 5)

if __name__ == "__main__":
    main()
