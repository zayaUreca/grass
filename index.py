import streamlit as st
from streamlit_folium import folium_static
from PIL import Image

# Config for website
st.set_page_config(
    page_title='Mongolia',
    layout="wide",
    initial_sidebar_state="expanded")

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

import streamlit as st
import geopandas as gpd
from shapely.geometry import Polygon
import folium
import mysql.connector

import cv2
import pandas as pd
import mysql.connector
# from datetime import datetime
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import datetime
import http.client
import json
import math
import requests
import os
from PIL import Image

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
    "database": "mongolia"
}

temp_range_low = -10
temp_range_high = 0

humidity_range_low = 20
humidity_range_high = 40

ndvi_range_low = -0.2
ndvi_range_high = 0.2

green_range_low = 33
green_range_high = 34.5

soil_t10_range_low = -1
soil_t10_range_high = 5

soil_moisture_range_low = -4
soil_moisture_range_high = 2



def count_pixels(image_path):
    # Open the image
    img = Image.open(image_path)
    # Convert the image to RGB mode
    img_rgb = img.convert("RGB")

    # Initialize counters
    green_count = 0
    yellow_count = 0
    red_count = 0

    # Iterate over each pixel in the image
    width, height = img.size
    for y in range(height):
        for x in range(width):
            # Get the RGB values of the pixel
            r, g, b = img_rgb.getpixel((x, y))

            # Check for green, yellow, and red pixels
            if g > r and g > b:  # Green pixel
                green_count += 1
            elif r > 100 and g > 100 and b < 50:  # Improved yellow pixel condition
            # elif r > g and g > b:
                yellow_count += 1
            # elif r > g and r > b:  # Red pixel
            elif r > g and r > b:
                red_count += 1

    return green_count, yellow_count, red_count


def calculate_greenness_percentage(image_path):
    # Load the image
    print("IMAGE PATH::::", image_path)
    image = cv2.imread(image_path)

    # Check if image is loaded properly
    if image is None:
        return 0

    # Split the image into RGB channels
    B, G, R = cv2.split(image)

    # Calculate the sum of RGB components for each pixel
    sum_RGB = R.astype("float") + G.astype("float") + B.astype("float")

    # To avoid division by zero, we'll add a small value (epsilon)
    epsilon = 1e-6
    sum_RGB += epsilon

    # Calculate the ratio of the Green component to the sum of RGB components
    ratio_green = G / sum_RGB

    # Calculate the average ratio of the green channel over the image
    average_green_ratio = np.mean(ratio_green)

    # Convert the ratio to percentage
    green_percentage = average_green_ratio * 100

    return green_percentage


def ndvi_parser(lon, lat):
    conn = http.client.HTTPSConnection("api.ambeedata.com")

    headers = {
        'x-api-key': "6474a5453a737994dbb5bf06c76356e4d18e0188fa554ad27457be1eaaea2779",
        'Content-type': "application/json"
    }

    conn.request("GET", f"/ndvi/latest/by-lat-lng?lat={lat}&lng={lon}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    response_dict = json.loads(data.decode("utf-8"))

    # Extract the NDVI value
    ndvi_value = response_dict["data"][0]["ndvi"]
    return ndvi_value


def fetch_weather_data(lat, lon, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = end_date - datetime.timedelta(days=7)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    # print("end_date:::::", end_date.strftime('%Y-%m-%d'))

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "is_day"]
    }
    responses = openmeteo.weather_api(url, params=params)
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["is_day"] = hourly_is_day

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    # print(hourly_dataframe)
    daytime_df = hourly_dataframe[hourly_dataframe['is_day'] == 1.0]

    # Calculate the mean temperature and humidity during the day
    mean_temperature = daytime_df['temperature_2m'].mean()
    mean_humidity = daytime_df['relative_humidity_2m'].mean()
    return mean_temperature, mean_humidity


def get_soil_data(lat, lon):
    api_key = 'f78ceda4749c6d73414fccce4962386c'
    url = 'http://api.agromonitoring.com/agro/1.0/soil'
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        soil_data = response.json()
        return soil_data["t0"] - 273.15, soil_data["moisture"]
    else:
        print("Failed to retrieve data:", response.status_code)
        return 0, 0


m = folium.Map(location=[46.8625, 103.8467], zoom_start=5)


# Assume previous initialization for Open-Meteo, database connection, etc., are done outside the function
# Define a function to save the uploaded image to a specific folder
def save_uploaded_file(uploaded_file, folder='uploaded_images'):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a file path
    file_path = os.path.join(folder, uploaded_file.name)

    # Write the uploaded file to the new file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path  # Return the file path of the saved image


def save_uploaded_file_ndvi(uploaded_file, folder='uploaded_images_ndvi'):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a file path
    file_path = os.path.join(folder, str(datetime.datetime.today().strftime('%Y-%m-%d')) + "_" + uploaded_file.name)

    # Write the uploaded file to the new file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path  # Return the file path of the saved image

def save_uploaded_file_herb(uploaded_file, herb_name, herb_lon, herb_lat, folder='uploaded_images_herb'):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a file path
    file_path = os.path.join(folder, f"{herb_name}_{herb_lon}_{herb_lat}_{uploaded_file.name}")

    # Write the uploaded file to the new file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def update_colors(m):
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    # SQL query
    query = "SELECT x1, y1, x2, y2, has_image, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10, yellow, red, green FROM my_table"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df = pd.read_sql(query, connection)

    # Convert coordinates to Polygon geometries and include fillColor
    df['geometry'] = df.apply(lambda row: Polygon([(row['x1'], row['y1']),
                                                   (row['x1'], row['y2']),
                                                   (row['x2'], row['y2']),
                                                   (row['x2'], row['y1'])]), axis=1)
    gdf2 = gpd.GeoDataFrame(df, geometry='geometry')

    temp_arr = df["temp"].to_numpy()
    humidity_arr = df["humidity"].to_numpy()
    color_arr = []
    soil_t10 = df["soil_t10"].to_numpy()
    ndvi_index = df["ndvi_index"].to_numpy()
    soil_moisture = df["soil_moisture"].to_numpy()
    green_score = df["green_score"].to_numpy()
    has_image = df["has_image"].to_numpy()
    green_ndvi = df["green"].to_numpy()
    red_ndvi = df["red"].to_numpy()
    yellow_ndvi = df["yellow"].to_numpy()
    # calculate color
    for i in range(len(has_image)):
        # print(has_image[i]==1)
        score = 0
        if has_image[i] == 1:
            if float(ndvi_index[i]) >= float(ndvi_range_high):
                score += 2
            elif float(ndvi_index[i]) >= float(ndvi_range_low):
                score += 1

            if float(temp_arr[i]) >= float(temp_range_high):
                score += 2
            elif float(temp_arr[i]) >= float(temp_range_low):
                score += 1

            if float(humidity_arr[i]) >= float(humidity_range_high):
                score += 2
            elif float(humidity_arr[i]) >= float(humidity_range_low):
                score += 1

            if float(green_score[i]) >= float(green_range_high):
                score += 2
            elif float(green_score[i]) >= float(green_range_low):
                score += 1

            if float(soil_t10[i]) >= float(soil_t10_range_high):
                score += 2
            elif float(soil_t10[i]) >= float(soil_t10_range_low):
                score += 1

            if float(soil_moisture[i]) >= float(soil_moisture_range_high):
                score += 2
            elif float(soil_moisture[i]) >= float(soil_moisture_range_low):
                score += 1

            if score >= 10:
                color_arr.append("#99FF99")
            elif score <= 2:
                color_arr.append("#FF0000")
            elif score > 2 and score < 4:
                color_arr.append("#FF6666")
            elif score > 7 and score < 10:
                color_arr.append("#99FF99")
            else:
                color_arr.append("#FFFF00")

        else:
            color_arr.append("#808080")

    for idx, row in gdf2.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        current_color = color_arr[idx]
        geo_j = folium.GeoJson(data=geo_j,
                               style_function=lambda x, idx=idx: {
                                   'fillColor': color_arr[idx],
                                   'color': 'black',
                                   'weight': 0.1,
                                   'fillOpacity': 0.3
                               })

        # Generate popup text with weather data
        if current_color == "#808080":
            popup_text = f"No image in that area, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"
        else:
            popup_text = f"Mean Temp: {temp_arr[idx]:.2f}°C, Mean Humidity: {humidity_arr[idx]:.2f}%, Green score: {green_score[idx]}, soil t10: {soil_t10[idx]}, soil moisture: {soil_moisture[idx]}, ndvi index: {ndvi_index[idx]}, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"

        # Add the popup with the weather data
        folium.Popup(popup_text).add_to(geo_j)

        geo_j.add_to(m)
    query = "SELECT lon, lat, green, yellow, red, image_date, image_name FROM my_table_ndvi"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df2 = pd.read_sql(query, connection)
    marker_lon = df2["lon"].to_numpy()
    marker_lat = df2["lat"].to_numpy()
    marker_green = df2["green"].to_numpy()
    marker_yellow = df2["yellow"].to_numpy()
    marker_red = df2["red"].to_numpy()
    marker_image_date = df2["image_date"].to_numpy()
    marker_image_name = df2["image_name"].to_numpy()
    for i in range(len(marker_lon)):
        popup_text = f"Green: {marker_green[i]}, Red: {marker_red[i]}, Yellow: {marker_yellow[i]}, Date: {marker_image_date[i]}, filename: {marker_image_name[i]}"
        folium.Marker(
            location=[marker_lat[i], marker_lon[i]],
            popup=popup_text,
        ).add_to(m)

def update_map_colors(m, lon, lat, file_name):
    lon = float(lon)
    lat = float(lat)
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    query = "SELECT x1, y1, x2, y2, has_image, image_date, images_list, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10 FROM my_table"
    with mysql.connector.connect(**db_config) as connection:
        df_table = pd.read_sql(query, connection)
    # y = lat x = lon
    x1 = df_table["x1"].to_numpy()
    y1 = df_table["y1"].to_numpy()
    x2 = df_table["x2"].to_numpy()
    y2 = df_table["y2"].to_numpy()
    has_image = df_table["has_image"].to_numpy()
    images_list = df_table["images_list"].to_numpy()
    green_score = df_table["green_score"].to_numpy().astype('str')
    image_date = df_table["image_date"].to_numpy()
    temp = df_table["temp"].to_numpy()
    humidity = df_table["humidity"].to_numpy()
    ndvi_index = df_table["ndvi_index"].to_numpy()
    soil_moisture = df_table["soil_moisture"]
    soil_t10 = df_table["soil_t10"]

    for i in range(len(has_image)):
        if y1[i] <= lat <= y2[i] and x1[i] <= lon <= x2[i]:
            has_image[i] = True
            images_list[i] = ""
            green_score[i] = ""
            image_date[i] = datetime.datetime.today().strftime('%Y-%m-%d')
            temp[i] = None
            humidity[i] = None

    for i in range(len(has_image)):
        if y1[i] <= lat <= y2[i] and x1[i] <= lon <= x2[i]:
            images_list[i] += f" {file_name}"
            # green_score[i] += f" {random.randint(0, 10)}"
            green_score[i] += f" {calculate_greenness_percentage(os.path.join(file_name))}"
            if temp[i] is None or math.isnan(float(temp[i])):
                cur_temp, cur_humidity = fetch_weather_data(lat, lon, datetime.datetime.today())
                cur_soil_moisture, cur_soil_t10 = get_soil_data(lat, lon)
                temp[i] = cur_temp
                humidity[i] = cur_humidity
                ndvi_index[i] = ndvi_parser(lon, lat)
                soil_moisture[i] = cur_soil_moisture
                soil_t10[i] = cur_soil_t10

    # update sql table
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Iterate over the DataFrame and update each row in the database
    for i in range(len(has_image)):
        update_command = (
            "UPDATE my_table SET has_image = %s, images_list = %s, green_score = %s, image_date = %s, temp = %s, humidity = %s, ndvi_index = %s, soil_moisture = %s, soil_t10 = %s WHERE id = %s")
        if image_date[i] == datetime.datetime.today().strftime('%Y-%m-%d'):
            new_green_index = np.mean([float(i) for i in green_score[i].split()])
            data = (has_image[i], images_list[i], new_green_index, image_date[i], float(temp[i]), float(humidity[i]),
                    ndvi_index[i], soil_moisture[i], soil_t10[i], i)
            cursor.execute(update_command, data)
    # Commit the changes to the database
    connection.commit()
    # Close the cursor and connection
    cursor.close()
    connection.close()

    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    # SQL query
    query = "SELECT x1, y1, x2, y2, has_image, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10, yellow, green, red FROM my_table"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df = pd.read_sql(query, connection)

    # Convert coordinates to Polygon geometries and include fillColor
    df['geometry'] = df.apply(lambda row: Polygon([(row['x1'], row['y1']),
                                                   (row['x1'], row['y2']),
                                                   (row['x2'], row['y2']),
                                                   (row['x2'], row['y1'])]), axis=1)
    gdf2 = gpd.GeoDataFrame(df, geometry='geometry')

    temp_arr = df["temp"].to_numpy()
    humidity_arr = df["humidity"].to_numpy()
    color_arr = []
    soil_t10 = df["soil_t10"].to_numpy()
    ndvi_index = df["ndvi_index"].to_numpy()
    soil_moisture = df["soil_moisture"].to_numpy()
    green_score = df["green_score"].to_numpy()
    has_image = df["has_image"].to_numpy()
    green_ndvi = df["green"].to_numpy()
    red_ndvi = df["red"].to_numpy()
    yellow_ndvi = df["yellow"].to_numpy()
    # calculate color
    for i in range(len(has_image)):
        # print(has_image[i]==1)
        score = 0
        if has_image[i] == 1:
            if float(ndvi_index[i]) >= float(ndvi_range_high):
                score+=2
            elif float(ndvi_index[i]) >= float(ndvi_range_low):
                score+=1

            if float(temp_arr[i]) >= float(temp_range_high):
                score+=2
            elif float(temp_arr[i]) >= float(temp_range_low):
                score+=1

            if float(humidity_arr[i]) >= float(humidity_range_high):
                score+=2
            elif float(humidity_arr[i]) >= float(humidity_range_low):
                score+=1

            if float(green_score[i]) >= float(green_range_high):
                score+=2
            elif float(green_score[i]) >= float(green_range_low):
                score+=1

            if float(soil_t10[i]) >= float(soil_t10_range_high):
                score+=2
            elif float(soil_t10[i]) >= float(soil_t10_range_low):
                score+=1

            if float(soil_moisture[i]) >= float(soil_moisture_range_high):
                score+=2
            elif float(soil_moisture[i]) >=float( soil_moisture_range_low):
                score+=1

            if score>=10:
                color_arr.append("#99FF99")
            elif score<=2:
                color_arr.append("#FF0000")
            elif score>2 and score<4:
                color_arr.append("#FF6666")
            elif score>7 and score<10:
                color_arr.append("#99FF99")
            else:
                color_arr.append("#FFFF00")
        else:
            color_arr.append("#808080")
    for idx, row in gdf2.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        current_color = color_arr[idx]
        geo_j = folium.GeoJson(data=geo_j,
                               style_function=lambda x, idx=idx: {
                                   'fillColor': color_arr[idx],
                                   'color': 'black',
                                   'weight': 0.1,
                                   'fillOpacity': 0.3
                               }).add_to(m)

        # Generate popup text with weather data
        if current_color=="#808080":
            popup_text = f"No image in that area, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"
        else:
            popup_text = f"Mean Temp: {temp_arr[idx]:.2f}°C, Mean Humidity: {humidity_arr[idx]:.2f}%, Green score: {green_score[idx]}, soil t10: {soil_t10[idx]}, soil moisture: {soil_moisture[idx]}, ndvi index: {ndvi_index[idx]}, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"

        # Add the popup with the weather data
        folium.Popup(popup_text).add_to(geo_j)

        geo_j.add_to(m)
    query = "SELECT lon, lat, green, yellow, red, image_date, image_name FROM my_table_ndvi"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df2 = pd.read_sql(query, connection)
    marker_lon = df2["lon"].to_numpy()
    marker_lat = df2["lat"].to_numpy()
    marker_green = df2["green"].to_numpy()
    marker_yellow = df2["yellow"].to_numpy()
    marker_red = df2["red"].to_numpy()
    marker_image_date = df2["image_date"].to_numpy()
    marker_image_name = df2["image_name"].to_numpy()
    for i in range(len(marker_lon)):
        popup_text = f"Green: {marker_green[i]}, Red: {marker_red[i]}, Yellow: {marker_yellow[i]}, Date: {marker_image_date[i]}, filename: {marker_image_name[i]}"
        folium.Marker(
            location=[marker_lat[i], marker_lon[i]],
            popup=popup_text,
        ).add_to(m)


def update_map_colors_ndvi(m, lon, lat, file_name):
    lon = float(lon)
    lat = float(lat)
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    query = "SELECT x1, y1, x2, y2, has_image, image_date, images_list, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10, yellow, green, red FROM my_table"
    with mysql.connector.connect(**db_config) as connection:
        df_table = pd.read_sql(query, connection)
    # y = lat x = lon
    x1 = df_table["x1"].to_numpy()
    y1 = df_table["y1"].to_numpy()
    x2 = df_table["x2"].to_numpy()
    y2 = df_table["y2"].to_numpy()

    green_ndvi = df_table["green"].to_numpy()
    red_ndvi = df_table["red"].to_numpy()
    yellow_ndvi = df_table["yellow"].to_numpy()
    for jj in range(len(y1)):
        if (y1[jj] <= lat <= y2[jj]) and (x1[jj] <= lon <= x2[jj]):
            green_count, yellow_count, red_count = count_pixels(file_name)

            query = ("INSERT INTO my_table_ndvi (lon, lat, green, yellow, red, image_date, image_name) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s)")
            data = (lon, lat, green_count, yellow_count, red_count, datetime.datetime.today().strftime('%Y-%m-%d'), file_name)
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            cursor.execute(query, data)
            connection.commit()
            cursor.close()
            connection.close()



            if green_ndvi[jj] is None or np.isnan(green_ndvi[jj]) or green_ndvi[jj]=="":
                green_ndvi[jj] = green_count
            else:
                green_ndvi[jj] = green_ndvi[jj] + green_count

            if yellow_ndvi[jj] is None or np.isnan(yellow_ndvi[jj]) or yellow_ndvi[jj]=="":
                yellow_ndvi[jj] = yellow_count
            else:
                yellow_ndvi[jj] = yellow_ndvi[jj] + yellow_count

            if red_ndvi[jj] is None or np.isnan(red_ndvi[jj]) or red_ndvi[jj]=="":
                red_ndvi[jj] = red_count
            else:
                red_ndvi[jj] = red_ndvi[jj] + red_count


    # update sql table
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    # Iterate over the DataFrame and update each row in the database
    for kk in range(len(y1)):
        if yellow_ndvi[kk] is not None and yellow_ndvi[kk]!="nan" and yellow_ndvi[kk]!="" and not np.isnan(yellow_ndvi[kk]):
            update_command = (
                "UPDATE my_table SET yellow = %s, green = %s, red = %s WHERE id = %s")
            data = (yellow_ndvi[kk], green_ndvi[kk], red_ndvi[kk], kk+1)
            cursor.execute(update_command, data)
    # Commit the changes to the database
    connection.commit()
    # Close the cursor and connection
    cursor.close()
    connection.close()

    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    # SQL query
    query = "SELECT x1, y1, x2, y2, has_image, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10, red, yellow, green FROM my_table"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df = pd.read_sql(query, connection)

    # Convert coordinates to Polygon geometries and include fillColor
    df['geometry'] = df.apply(lambda row: Polygon([(row['x1'], row['y1']),
                                                   (row['x1'], row['y2']),
                                                   (row['x2'], row['y2']),
                                                   (row['x2'], row['y1'])]), axis=1)
    gdf2 = gpd.GeoDataFrame(df, geometry='geometry')

    # colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

    temp_arr = df["temp"].to_numpy()
    humidity_arr = df["humidity"].to_numpy()
    color_arr = []
    soil_t10 = df["soil_t10"].to_numpy()
    ndvi_index = df["ndvi_index"].to_numpy()
    soil_moisture = df["soil_moisture"].to_numpy()
    green_score = df["green_score"].to_numpy()
    has_image = df["has_image"].to_numpy()
    green_ndvi = df["green"].to_numpy()
    red_ndvi = df["red"].to_numpy()
    yellow_ndvi = df["yellow"].to_numpy()
    # calculate color
    for i in range(len(has_image)):
        # print(has_image[i]==1)
        score = 0
        if has_image[i] == 1:
            if float(ndvi_index[i]) >= float(ndvi_range_high):
                score+=2
            elif float(ndvi_index[i]) >= float(ndvi_range_low):
                score+=1

            if float(temp_arr[i]) >= float(temp_range_high):
                score+=2
            elif float(temp_arr[i]) >= float(temp_range_low):
                score+=1

            if float(humidity_arr[i]) >= float(humidity_range_high):
                score+=2
            elif float(humidity_arr[i]) >= float(humidity_range_low):
                score+=1

            if float(green_score[i]) >= float(green_range_high):
                score+=2
            elif float(green_score[i]) >= float(green_range_low):
                score+=1

            if float(soil_t10[i]) >= float(soil_t10_range_high):
                score+=2
            elif float(soil_t10[i]) >= float(soil_t10_range_low):
                score+=1

            if float(soil_moisture[i]) >= float(soil_moisture_range_high):
                score+=2
            elif float(soil_moisture[i]) >=float( soil_moisture_range_low):
                score+=1

            if score>=10:
                color_arr.append("#99FF99")
            elif score<=2:
                color_arr.append("#FF0000")
            elif score>2 and score<4:
                color_arr.append("#FF6666")
            elif score>7 and score<10:
                color_arr.append("#99FF99")
            else:
                color_arr.append("#FFFF00")
        else:
            color_arr.append("#808080")
    for idx, row in gdf2.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        current_color = color_arr[idx]
        geo_j = folium.GeoJson(data=geo_j,
                               style_function=lambda x, idx=idx: {
                                   'fillColor': color_arr[idx],
                                   'color': 'black',
                                   'weight': 0.1,
                                   'fillOpacity': 0.3
                               }).add_to(m)

        # Generate popup text with weather data
        print( f"No {idx} image in that area, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}")
        if current_color=="#808080":
            popup_text = f"No image in that area, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"
        else:
            popup_text = f"Mean Temp: {temp_arr[idx]:.2f}°C, Mean Humidity: {humidity_arr[idx]:.2f}%, Green score: {green_score[idx]}, soil t10: {soil_t10[idx]}, soil moisture: {soil_moisture[idx]}, ndvi index: {ndvi_index[idx]}, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"
        # Add the popup with the weather data
        folium.Popup(popup_text).add_to(geo_j)

        geo_j.add_to(m)

    query = "SELECT lon, lat, green, yellow, red, image_date, image_name FROM my_table_ndvi"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df2 = pd.read_sql(query, connection)
    marker_lon = df2["lon"].to_numpy()
    marker_lat = df2["lat"].to_numpy()
    marker_green = df2["green"].to_numpy()
    marker_yellow = df2["yellow"].to_numpy()
    marker_red = df2["red"].to_numpy()
    marker_image_date = df2["image_date"].to_numpy()
    marker_image_name = df2["image_name"].to_numpy()
    for i in range(len(marker_lon)):
        popup_text = f"Green: {marker_green[i]}, Red: {marker_red[i]}, Yellow: {marker_yellow[i]}, Date: {marker_image_date[i]}, filename: {marker_image_name[i]}"
        folium.Marker(
            location=[marker_lat[i], marker_lon[i]],
            popup=popup_text,
        ).add_to(m)



with st.sidebar:
    # Load profiles as python dict.

    st.header('Profiles currently in our database')
    st.subheader('Click through them to learn about the region')

    # Lets me programatically build the sidebar based on data in profiles.
    country = st.selectbox(label='Country', label_visibility='collapsed',
                           options=('Mongolia', '-'))

    temp_range_low_in = st.text_input("temp low :", "")
    temp_range_high_in = st.text_input("temp high:", "")

    humidity_range_low_in = st.text_input("humidity low :", "")
    humidity_range_high_in = st.text_input("humidity high:", "")

    ndvi_range_low_in = st.text_input("ndvi low :", "")
    ndvi_range_high_in = st.text_input("ndvi high:", "")

    st.text('Recomended range for green: 33 - 34.5')
    green_range_low_in = st.text_input("green low :", "")
    green_range_high_in = st.text_input("green high:", "")


    soil_t10_range_low_in = st.text_input("soil_t10 low :", "")
    soil_t10_range_high_in = st.text_input("soil_t10 high:", "")

    soil_moisture_range_low_in = st.text_input("soil_moisture low :", "")
    soil_moisture_range_high_in = st.text_input("soil_moisture high:", "")

    if st.button('Update params'):

        temp_range_low = temp_range_low_in
        temp_range_high = temp_range_high_in
        humidity_range_low = humidity_range_low_in
        humidity_range_high = humidity_range_high_in
        ndvi_range_low = ndvi_range_low_in
        ndvi_range_high = ndvi_range_high_in
        green_range_low = green_range_low_in
        green_range_high = green_range_high_in
        soil_t10_range_low = soil_t10_range_low_in
        soil_t10_range_high = soil_t10_range_high_in
        soil_moisture_range_low = soil_moisture_range_low_in
        soil_moisture_range_high = soil_moisture_range_high_in
        st.text(
            f'Current parameters: temp({temp_range_low}-{temp_range_high}), humidity({humidity_range_low}-{humidity_range_high}), ndvi({ndvi_range_low}-{ndvi_range_high}), green({green_range_low}-{green_range_high}), soil_t10({soil_t10_range_low}-{soil_t10_range_high}), soil({soil_moisture_range_low}-{soil_moisture_range_high})')

        update_colors(m)


    uploaded_file = st.file_uploader("Add photo from location", type=["jpg", "jpeg", "png"])
    # Input for longitude and latitude
    longitude = st.text_input("Longitude:", "")
    latitude = st.text_input("Latitude:", "")

    # Button to process everything
    if st.button('Process Inputs'):
        if uploaded_file is not None and longitude and latitude:
            # Save the uploaded image
            file_path = save_uploaded_file(uploaded_file)
            # Display a message with the file path and coordinates
            update_map_colors(m, longitude, latitude, file_path)
        else:
            # Display an error message if not all inputs are provided
            st.error("Please upload an image and enter both longitude and latitude.")

    uploaded_file_ndvi = st.file_uploader("Add ndvi photo", type=["tif"])
    # Input for longitude and latitude
    longitude_ndvi = st.text_input("Longitude ndvi:", "")
    latitude_ndvi = st.text_input("Latitude ndvi:", "")

    # Button to process everything
    if st.button('Update ndvi'):
        if uploaded_file_ndvi is not None and longitude_ndvi and latitude_ndvi:
            # Save the uploaded image
            file_path_ndvi = save_uploaded_file_ndvi(uploaded_file_ndvi)

            # Display a message with the file path and coordinates
            update_map_colors_ndvi(m, longitude_ndvi, latitude_ndvi, file_path_ndvi)
        else:
            # Display an error message if not all inputs are provided
            st.error("Please upload an image and enter both longitude and latitude.")

st.title('Site Suitability for Mongolia Area')
st.write(
    "Select any location in Mongolia to see information about that square")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["The App", "How It Works", "Herbarium", "NDVI", "NDVI List"])




with tab3:
    st.title("Herbarium gallery")
    uploaded_file_herb = st.file_uploader("Add photo of herbarium", type=["jpg", "jpeg", "png"])
    herb_name = st.text_input("Enter herbarium name:", "")
    herb_lon = st.text_input("Enter herbarium longitude:", "")
    herb_lat = st.text_input("Enter herbarium latitude:", "")
    if st.button('Upload image'):
        if uploaded_file_herb is not None and herb_name and herb_lon and herb_lat:
            # Save the uploaded image
            file_path = save_uploaded_file_herb(uploaded_file_herb, herb_name, herb_lon, herb_lat)
            image = Image.open(file_path)
            herb_name_of_file = herb_name
        else:
            # Display an error message if not all inputs are provided
            st.error("Please upload an image and enter name")

    # List all files in the directory
    files = os.listdir("uploaded_images_herb")

    # Filter out files that are not images (optional, based on file extensions)
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

    # Display each image using Streamlit
    for image_file in image_files:
        image_path = os.path.join("uploaded_images_herb", image_file)
        image = Image.open(image_path)

        # Display the image
        herb_name_of_file = image_file.split("_")[0]
        herb_lon_of_file = image_file.split("_")[1]
        herb_lat_of_file = image_file.split("_")[2]
        cur_caption = f"name:{herb_name_of_file}, lon:{herb_lon_of_file}, lat:{herb_lat_of_file}"
        # st.text(herb_name_of_file)
        st.image(image, caption=cur_caption, use_column_width=True)


with tab2:
    st.title('The process will include the following steps:')
    st.write("""
    1.\tDisplay of sectors with greening: sectors will be marked with squares, the color of which changes depending on the level of greening, temperature, humidity, ndvi index, soil moisture and soil t10.
    \n2.\tAdding photos from the location: The user needs to add photos taken on the premises to assess the level of landscaping. After uploading a photo, the system classifies it by coordinates.
    \n3.\tTemperature and humidity data integration: Temperature and humidity data are integrated into the system for each sector. These data can be presented in the form of numerical values reflecting changes in temperature and humidity in different sectors.

    \n4.\tFIELD Data Integration: The Normalized Differential Vegetation Index can be used to estimate the condition and density of vegetation in sectors. This data is also integrated into the system and displayed on the map.



    \nIn this way, the system will provide a more complete picture of the condition of green spaces, including data on photos from the site, temperature, humidity, soil fertility and vegetation condition.

    """)
    st.write('')
    st.title('To use the platform correctly, you need:')
    st.write("""
    1. \tTo begin with, you will need to collect a photo with the location that will be analyzed. This photo can be taken from a drone at a height, or taken from the ground.
    (But for a correct assessment, it is recommended to take photos from drones)
    \n 2.\tIn the next step, you will need to register the norms in the "Profiles currently in our database" section. These rules are already written in the program, but you have the opportunity to make your own changes. Data that are already registered in the program are given in the same section.
    \n If necessary, you will need to enter these data:
    \n"temp low"
    \n"temp high"
    \n"low humidity"
    \n"humidity high"
    \n"ndvi low"
    \n"ndvi high"
    \n
    \n And also, if necessary, write down these data:
    \n"green low"
    \n"green high"
    \n"soil_t10 low"
    \n"soil_t10 high"
    \n"soil_moisture low"
    \n"soil_moisture high"
    \n 3.\tAfter you enter all the data that you will need to enter in one or another case, you will need to upload the photo mentioned in section "1".
    The photo should be in the format: "JPG", "JPEG", "PNG", and the size should not exceed 200MB.
    \n4.\tAfter adding the photo, you will need to write down the coordinates of the location that will be analyzed for correct addition to the map. Namely: "Longitude", "Latitude".
    According to these coordinates, the data to be analyzed will be added to the sector on the map. Sectors on the map are divided by size, each sector is 100 by 100 kilometers.
    """)




with tab1:
    # Load Mongolia boundary
    mongolia = gpd.read_file('gadm41_MNG_1.shp')
    mongolia_polygon = mongolia.unary_union

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Define the bounds of Mongolia (approximate for this example)
    xmin, ymin, xmax, ymax = 87.751264, 41.597409, 119.772824, 52.1496


    import mysql.connector

    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "cFdbU@hNFENAn!9_HnnBTXTdXmh@",
        "database": "mongolia"
    }

    # SQL query
    query = "SELECT x1, y1, x2, y2, has_image, green_score, temp, humidity, ndvi_index, soil_moisture, soil_t10, green, yellow, red FROM my_table"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df = pd.read_sql(query, connection)

    # Convert coordinates to Polygon geometries and include fillColor
    df['geometry'] = df.apply(lambda row: Polygon([(row['x1'], row['y1']),
                                                   (row['x1'], row['y2']),
                                                   (row['x2'], row['y2']),
                                                   (row['x2'], row['y1'])]), axis=1)
    gdf2 = gpd.GeoDataFrame(df, geometry='geometry')
    temp_arr = df["temp"].to_numpy()
    humidity_arr = df["humidity"].to_numpy()
    color_arr = []
    soil_t10 = df["soil_t10"].to_numpy()
    ndvi_index = df["ndvi_index"].to_numpy()
    soil_moisture = df["soil_moisture"].to_numpy()
    green_score = df["green_score"].to_numpy()
    has_image = df["has_image"].to_numpy()
    green_ndvi = df["green"].to_numpy()
    red_ndvi = df["red"].to_numpy()
    yellow_ndvi = df["yellow"].to_numpy()
    #calculate color
    for i in range(len(has_image)):
        # print(has_image[i]==1)
        score = 0
        if has_image[i]==1:
            # print("NDVI INDEX STR TO FLOAT::", ndvi_index[i])
            # print("NDVI INDEX STR TO FLOAT2::", ndvi_range_high)
            if float(ndvi_index[i]) >= float(ndvi_range_high):
                score += 2
            elif float(ndvi_index[i]) >= float(ndvi_range_low):
                score += 1

            if float(temp_arr[i]) >= float(temp_range_high):
                score += 2
            elif float(temp_arr[i]) >= float(temp_range_low):
                score += 1

            if float(humidity_arr[i]) >= float(humidity_range_high):
                score += 2
            elif float(humidity_arr[i]) >= float(humidity_range_low):
                score += 1

            if float(green_score[i]) >= float(green_range_high):
                score += 2
            elif float(green_score[i]) >= float(green_range_low):
                score += 1

            if float(soil_t10[i]) >= float(soil_t10_range_high):
                score += 2
            elif float(soil_t10[i]) >= float(soil_t10_range_low):
                score += 1

            if float(soil_moisture[i]) >= float(soil_moisture_range_high):
                score += 2
            elif float(soil_moisture[i]) >= float(soil_moisture_range_low):
                score += 1

            if score >= 10:
                color_arr.append("#99FF99")
            elif score <= 2:
                color_arr.append("#FF0000")
            elif score > 2 and score < 4:
                color_arr.append("#FF6666")
            elif score > 7 and score < 10:
                color_arr.append("#99FF99")
            else:
                color_arr.append("#FFFF00")

        else:
            color_arr.append("#808080")
    for idx, row in gdf2.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        current_color = color_arr[idx]
        geo_j = folium.GeoJson(data=geo_j,
                               style_function=lambda x, idx=idx: {
                                   'fillColor':  color_arr[idx],
                                   'color': 'black',
                                   'weight': 0.1,
                                   'fillOpacity': 0.3
                               })

        # Generate popup text with weather data
        if current_color=="#808080":
            popup_text = f"No image in that area, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"
        else:
            popup_text = f"Mean Temp: {temp_arr[idx]:.2f}°C, Mean Humidity: {humidity_arr[idx]:.2f}%, Green score: {green_score[idx]}, soil t10: {soil_t10[idx]}, soil moisture: {soil_moisture[idx]}, ndvi index: {ndvi_index[idx]}, Green: {green_ndvi[idx]}, Red: {red_ndvi[idx]}, Yellow: {yellow_ndvi[idx]}"


        # Add the popup with the weather data
        folium.Popup(popup_text).add_to(geo_j)

        geo_j.add_to(m)
    query = "SELECT lon, lat, green, yellow, red, image_date, image_name FROM my_table_ndvi"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df2 = pd.read_sql(query, connection)
    marker_lon = df2["lon"].to_numpy()
    marker_lat = df2["lat"].to_numpy()
    marker_green = df2["green"].to_numpy()
    marker_yellow = df2["yellow"].to_numpy()
    marker_red = df2["red"].to_numpy()
    marker_image_date = df2["image_date"].to_numpy()
    marker_image_name = df2["image_name"].to_numpy()
    for i in range(len(marker_lon)):
        popup_text = f"Green: {marker_green[i]}, Red: {marker_red[i]}, Yellow: {marker_yellow[i]}, Date: {marker_image_date[i]}, filename: {marker_image_name[i]}"
        folium.Marker(
            location=[marker_lat[i], marker_lon[i]],
            popup=popup_text,
        ).add_to(m)

    # Display the map in Streamlit
    folium_static(m, width=1000)


with tab4:
    st.title("NDVI gallery")
    files = os.listdir("uploaded_images_ndvi")
    image_files = [file for file in files if file.endswith(('.tif'))]
    for image_file in image_files:
        image_path = os.path.join("uploaded_images_ndvi", image_file)
        image = Image.open(image_path)
        cur_caption = f"name:{image_file}"
        st.image(image, caption=cur_caption, use_column_width=True)


with tab5:
    st.title("NDVI list")
    query = "SELECT lon, lat, green, yellow, red, image_date, image_name FROM my_table_ndvi"
    # Fetch the data
    with mysql.connector.connect(**db_config) as connection:
        df2 = pd.read_sql(query, connection)
    st.table(df2)

