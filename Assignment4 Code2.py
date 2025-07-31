import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import tkinter as tk
from tkinter import filedialog

# ====================
# Data Loading using Tkinter
# ====================
root = tk.Tk()
root.withdraw()  # Hide the root Tk window

# Define file keys for CSV files.
'''
files = {
    "latitude": "C:\\Users\\Anonym\\Downloads\\eMaintanceAssignment4\\data2\\2024-12-08 02-00-00 (1)\\2024-12-08 02-00-00 (1)\\GPS.latitude.csv", #need tofill in here
    "longitude": "C:\\Users\\Anonym\\Downloads\\eMaintanceAssignment4\\data2\\2024-12-08 02-00-00 (1)\\2024-12-08 02-00-00 (1)\\GPS.longitude.csv",
    "vibration1": "C:\\Users\\Anonym\\Downloads\\eMaintanceAssignment4\\data2\\2024-12-08 02-00-00 (1)\\2024-12-08 02-00-00 (1)\\GPS.CH1_ACCEL1Z1.csv",
    "vibration2": "C:\\Users\\Anonym\\Downloads\\eMaintanceAssignment4\\data2\\2024-12-08 02-00-00 (1)\\2024-12-08 02-00-00 (1)\\GPS.CH2_ACCEL1Z2.csv",
    "speed": "C:\\Users\\Anonym\\Downloads\\eMaintanceAssignment4\\data2\\2024-12-08 02-00-00 (1)\\2024-12-08 02-00-00 (1)\\GPS.speed.csv"
}
'''
files = {
    "latitude": None, #need tofill in here
    "longitude": None,
    "vibration1":None,
    "vibration2":None,
    "speed": None
}
def load_file(key):
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        files[key] = file_path
        print(f"{key.capitalize()} file loaded: {file_path}")





print("Select Latitude File")
load_file("latitude")
print("Select Longitude File")
load_file("longitude")
print("Select Vibration 1 File")
#load_file("vibration1")
print("Select Vibration 2 File")
#load_file("vibration2")
print("Select Speed File")
#load_file("speed")


# Load each CSV into a DataFrame and add a 'timestamp' using the row index.
dataframes = {}
for key, file_path in files.items():
    if file_path:
        df = pd.read_csv(file_path, header=None, names=[key])
        df['timestamp'] = df.index
        dataframes[key] = df
    else:
        print(f"{key.capitalize()} file not selected.")

# ====================
# Create GPS DataFrame by merging latitude and longitude.
# ====================
if "latitude" in dataframes and "longitude" in dataframes:
    df_gps = pd.merge(dataframes["latitude"], dataframes["longitude"], on="timestamp")
    # Rename columns for consistency
    df_gps = df_gps.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    # Add an index column for use in the interactive plot
    df_gps["PointIndex"] = df_gps.index
else:
    print("Latitude or Longitude data is missing.")
    df_gps = pd.DataFrame(columns=["Latitude", "Longitude", "PointIndex"])

# ====================
# Merge the two vibration signals on 'timestamp'
# ====================
if "vibration1" in dataframes and "vibration2" in dataframes:
    df_vibration_merged = pd.merge(
        dataframes["vibration1"],
        dataframes["vibration2"],
        on="timestamp"
        # When the column names differ (here: vibration1 vs vibration2), suffixes are not needed.
    )
    # You may rename columns if desired; here they remain "vibration1" and "vibration2"
else:
    print("Vibration data files are missing.")
    df_vibration_merged = pd.DataFrame()



# ====================
# Data Preprocessing and Segmentation for Vibration Data
# ====================
dt_vibration = 0.002  # seconds per sample (e.g. 500 Hz sampling rate)
segment_duration_seconds = 10
segment_length = int(segment_duration_seconds / dt_vibration)
if not df_vibration_merged.empty:
    num_segments = len(df_vibration_merged) // segment_length
    segments = []
    for i in range(num_segments):
        seg = df_vibration_merged.iloc[i * segment_length: (i + 1) * segment_length][["vibration1", "vibration2"]].values
        segments.append(seg)
    segments = np.array(segments)
    print("Segmented vibration data shape:", segments.shape)
else:
    segments = np.array([])
    print("No vibration data available for segmentation.")


# ====================
# Build the Interactive Dash App
# ====================

# Create the interactive GPS map using Plotly Express.
if not df_gps.empty:
    # Use custom_data to store the point index so that it will be available in callbacks.
    map_fig = px.scatter_mapbox(
        df_gps,
        lat="Latitude",
        lon="Longitude",
        custom_data=["PointIndex"],
        zoom=10,
        title="GPS Points with Vibration Data"
    )
    map_fig.update_layout(mapbox_style="open-street-map", height=600)
else:
    map_fig = go.Figure()
    map_fig.update_layout(title="No GPS Data Available", height=600)

# Create an initial empty vibration plot figure.
vib_empty_fig = go.Figure()
vib_empty_fig.update_layout(
    title="Vibration Signal",
    xaxis_title="Time (s)",
    yaxis_title="Acceleration"
)

# Initialize Dash app.
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="gps-map", figure=map_fig)
    ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
    html.Div([
        dcc.Graph(id="vibration-plot", figure=vib_empty_fig)
    ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
])

# --------------------
# Callback to Update the Vibration Plot Based on Clicked GPS Point
# --------------------
@app.callback(
    Output('vibration-plot', 'figure'),
    [Input('gps-map', 'clickData')]
)
def update_vibration_plot(clickData):
    # If no point is selected, return the empty vibration figure.
    if clickData is None:
        return vib_empty_fig

    # Retrieve the selected GPS point index from custom_data.
    point_index = clickData['points'][0]['pointIndex']


    if segments.size == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Vibration Data Available",
            xaxis_title="Time (s)",
            yaxis_title="Acceleration"
        )
        return empty_fig

    # Map the GPS point to a vibration segment. If the selected index exceeds
    # the available number of segments, use the last segment as fallback.
    if point_index < segments.shape[0]:
        selected_segment = segments[point_index]
    else:
        selected_segment = segments[-1]

    # Create a time axis for the selected segment.
    time_axis = np.arange(segment_length) * dt_vibration

    vib_fig = go.Figure()
    vib_fig.add_trace(go.Scatter(
        x=time_axis,
        y=selected_segment[:, 0],
        mode='lines',
        name='Vibration Channel 1'
    ))
    vib_fig.add_trace(go.Scatter(
        x=time_axis,
        y=selected_segment[:, 1],
        mode='lines',
        name='Vibration Channel 2'
    ))
    vib_fig.update_layout(
        title=f"Vibration Signal for GPS Point {point_index}",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration"
    )
    return vib_fig

# ====================
# Run the Dash App
# ====================
if __name__ == "__main__":
    print("starting")
    app.run(debug=True, port=8060)