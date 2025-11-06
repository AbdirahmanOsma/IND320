from __future__ import annotations

import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram

load_dotenv()  # local dev fallback

def _get_mongo_uri() -> str | None:
    # Streamlit Cloud first
    try:
        return st.secrets["MONGODB_URI"]
    except Exception:
        pass
    # Local .env fallback
    return os.getenv("MONGODB_URI")

MONGODB_URI = _get_mongo_uri()
if not MONGODB_URI:
    st.error("Missing MONGODB_URI. Set it in Streamlit Secrets (Cloud) or .env (local).")
    st.stop()

# Debug line - show which database you're connecting to
st.caption("Mongo host: " + MONGODB_URI.split("@")[-1])

st.set_page_config(page_title="IND320 - Dashboard (Part 1 + 2 + 3)", layout="wide")

# --- Weather API Function ---
@st.cache_data
def download_weather_data(latitude, longitude, year=2021):
    """Download historical weather data from open-meteo.com for 2021"""
    url = (
        "https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={year}-01-01"
        f"&end_date={year}-12-31"
        "&hourly=temperature_2m,precipitation,relative_humidity_2m,"
        "pressure_msl,wind_speed_10m"
        "&timezone=Europe%2FOslo"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

# City coordinates for price areas
cities = {
    "NO1": {"city": "Oslo", "lat": 59.9139, "lon": 10.7522},
    "NO2": {"city": "Kristiansand", "lat": 58.1467, "lon": 7.9956},
    "NO3": {"city": "Trondheim", "lat": 63.4305, "lon": 10.3951},
    "NO4": {"city": "Tromsø", "lat": 69.6492, "lon": 18.9553},
    "NO5": {"city": "Bergen", "lat": 60.3930, "lon": 5.3242},
}

# --- Analysis Functions ---
def plot_temperature_outliers(df, temp_col="temperature_2m", time_col="time", 
                             cutoff_freq=0.02, n_sigma=3.0):
    d = df[[time_col, temp_col]].copy()
    d = d.dropna(subset=[temp_col]).drop_duplicates(subset=[time_col]).sort_values(time_col)
    d[time_col] = pd.to_datetime(d[time_col], utc=True)
    d[temp_col] = d[temp_col].interpolate(limit_direction="both")
    
    x = d[temp_col].to_numpy()
    X = dct(x, norm="ortho")
    n = len(X)
    cut_idx = int(np.clip(cutoff_freq * n, 0, n-1))
    X[:cut_idx] = 0.0
    satv = idct(X, norm="ortho")
    
    med = np.median(satv)
    mad = np.median(np.abs(satv - med))
    robust_sd = 1.4826 * mad if mad > 0 else np.std(satv)
    upper = med + n_sigma * robust_sd
    lower = med - n_sigma * robust_sd
    outlier_mask = (satv > upper) | (satv < lower)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(d[time_col], d[temp_col], label="Temperature", linewidth=1)
    if outlier_mask.any():
        ax.scatter(d.loc[outlier_mask, time_col], d.loc[outlier_mask, temp_col], 
                  color="red", label="Outliers", s=20)
    ax.axhline(d[temp_col].mean() + (upper-med), linestyle="--", color="green", label="Upper limit")
    ax.axhline(d[temp_col].mean() + (lower-med), linestyle="--", color="orange", label="Lower limit")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Outlier Detection - DCT + SPC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    summary = {
        "n_points": len(d),
        "n_outliers": outlier_mask.sum(),
        "outlier_percent": round(100 * outlier_mask.mean(), 2),
        "robust_sd": robust_sd,
    }
    return fig, summary

def detect_precipitation_anomalies(df, precip_col="precipitation", time_col="time", contamination=0.01):
    d = df[[time_col, precip_col]].copy()
    d = d.dropna(subset=[precip_col])
    d[time_col] = pd.to_datetime(d[time_col], utc=True)
    d = d.sort_values(time_col)
    d["time_index"] = (d[time_col] - d[time_col].min()).dt.total_seconds() / 3600
    
    X = d[[precip_col, "time_index"]].to_numpy()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = lof.fit_predict(X)
    anomaly_mask = (labels == -1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(d[time_col], d[precip_col], label="Precipitation", linewidth=1)
    if anomaly_mask.any():
        ax.scatter(d.loc[anomaly_mask, time_col], d.loc[anomaly_mask, precip_col], 
                  color="red", label="Anomalies", s=20)
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title("Precipitation Anomaly Detection - LOF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    summary = {
        "n_points": len(d),
        "n_anomalies": anomaly_mask.sum(),
        "anomaly_percent": round(100 * anomaly_mask.mean(), 2),
    }
    return fig, summary

def stl_decompose_elhub(df, price_area="NO5", production_group="hydro",
                        period=7, seasonal=13, trend=101, robust=True):
    daily_data = (df[(df["priceArea"] == price_area) &
                     (df["productionGroup"] == production_group)]
                .set_index("startTime")[["quantitykwh"]]
                .resample("D").sum()
                .rename(columns={"quantitykwh": "kWh"}))
    
    series = daily_data["kWh"].astype(float)
    stl_result = STL(series, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(series.index, series.values, color='blue')
    axes[0].set_title(f'Original - {price_area}/{production_group}')
    axes[0].set_ylabel('kWh')
    axes[0].grid(alpha=0.3)

    axes[1].plot(series.index, stl_result.trend, color='green')
    axes[1].set_title('Trend')
    axes[1].set_ylabel('kWh')
    axes[1].grid(alpha=0.3)

    axes[2].plot(series.index, stl_result.seasonal, color='orange')
    axes[2].set_title('Seasonal')
    axes[2].set_ylabel('kWh')
    axes[2].grid(alpha=0.3)

    axes[3].plot(series.index, stl_result.resid, color='red')
    axes[3].set_title('Residual')
    axes[3].set_ylabel('kWh')
    axes[3].set_xlabel('Date')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    return fig

def plot_production_spectrogram(df, price_area="NO5", production_group="hydro",
                                window_length=24*7, overlap=0.5):
    filtered = df[(df["priceArea"] == price_area) &
                  (df["productionGroup"] == production_group)].copy()
    filtered["startTime"] = pd.to_datetime(filtered["startTime"], utc=True)
    hourly_data = filtered.set_index("startTime")["quantitykwh"].resample("H").mean()
    hourly_data = hourly_data.interpolate()

    nperseg = window_length
    noverlap = int(nperseg * overlap)
    f, t, Sxx = spectrogram(hourly_data.values, fs=1/3600, nperseg=nperseg, noverlap=noverlap)

    fig, ax = plt.subplots(figsize=(12, 6))
    pcm = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-8), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [hours]')
    ax.set_title(f'Spectrogram - {price_area}/{production_group}')
    plt.colorbar(pcm, ax=ax).set_label('Power Spectral Density [dB]')
    return fig

# --- Sample Data Fallback ---
@st.cache_data
def get_sample_energy_data():
    """Return sample data for demonstration when MongoDB fails"""
    dates = pd.date_range('2021-01-01', '2021-01-10', freq='D')
    sample_data = []
    
    for area in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']:
        for group in ['hydro', 'wind', 'solar', 'thermal', 'other']:
            for date in dates:
                sample_data.append({
                    'priceArea': area,
                    'productionGroup': group,
                    'quantitykwh': np.random.randint(1000, 10000),
                    'startTime': date
                })
    
    df_sample = pd.DataFrame(sample_data)
    df_sample["startTime"] = pd.to_datetime(df_sample["startTime"])
    return df_sample

# --- MongoDB Connection ---
@st.cache_resource
def get_energy_data():
    try:
        st.info("Connecting to MongoDB...")
        
        # Use environment variable for MongoDB connection
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        client.admin.command('ping')
        st.success("Connected to MongoDB Atlas!")
        
        database = client["ind320"]
        collection = database["elhub_production_data_2021"]
        data = list(collection.find({}, {"_id": 0}))
        df_energy = pd.DataFrame(data)
        
        column_mapping = {}
        for col_name in df_energy.columns:
            if col_name.lower() == "pricearea":
                column_mapping[col_name] = "priceArea"
            elif col_name.lower() == "productiongroup":
                column_mapping[col_name] = "productionGroup"
            elif col_name.lower() == "starttime":
                column_mapping[col_name] = "startTime"
            elif col_name.lower() == "quantitykwh":
                column_mapping[col_name] = "quantitykwh"
        
        df_energy.rename(columns=column_mapping, inplace=True)
        df_energy["startTime"] = pd.to_datetime(df_energy["startTime"], utc=True, errors="coerce")
        df_energy["quantitykwh"] = pd.to_numeric(df_energy["quantitykwh"], errors="coerce")
        df_energy = df_energy.dropna(subset=["priceArea", "productionGroup", "startTime", "quantitykwh"]).sort_values("startTime")
        
        return df_energy
    except Exception as e:
        st.error(f"Database error: {e}")
        # Fallback to sample data
        return get_sample_energy_data()

# --- Sidebar Navigation ---
st.sidebar.title("IND320 Navigation")
st.sidebar.markdown("---")

pages = [
    "Home",
    "Data Table", 
    "Plots",
    "Mongo Dashboard",
    "STL / Spectrogram (Part 3A)",
    "Outlier / Anomaly (Part 3B)",
    "About"
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("**IND320 Dashboard = Parts 1, 2 & 3**")

# --- Page 1: Home ---
if page == "Home":
    st.title("IND320 - Dashboard (Part 1 + 2 + 3)")
    st.markdown("Use the sidebar to navigate between pages.")
    st.markdown("---")
    
    st.subheader("Welcome")
    st.markdown("""
    This dashboard demonstrates the progression of IND320 Part 1, 2 and 3:
    
    - **Part 1**: CSV loaded locally — Data summary and plots
    - **Part 2**: MongoDB connection — Interactive Elhub dashboard  
    - **Part 3**: Open Meteo API — STL, Spectrogram & Outlier Detection
    """)
    
    st.markdown("---")
    
    # Quick data preview
    st.subheader("Quick Data Preview")
    selected_area = st.selectbox("Select Price Area for Weather Data", list(cities.keys()))
    
    if st.button("Load 2021 Weather Data"):
        with st.spinner("Downloading data..."):
            try:
                city = cities[selected_area]
                weather_df = download_weather_data(city["lat"], city["lon"], 2021)
                st.session_state.weather_data = weather_df
                st.session_state.selected_area = selected_area
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", weather_df.shape[0])
                col2.metric("Columns", weather_df.shape[1])
                col3.metric("Area", f"{city['city']} ({selected_area})")
                
                st.dataframe(weather_df.head())
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- Page 2: Data Table ---
elif page == "Data Table":
    st.header("Data Table")
    
    if 'weather_data' in st.session_state:
        df = st.session_state.weather_data
        NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
        
        first_month = df["time"].dt.to_period("M").min()
        mask = df["time"].dt.to_period("M") == first_month
        df_first = df.loc[mask, ["time"] + NUMERIC_COLS]

        rows = []
        for col in NUMERIC_COLS:
            rows.append({"variable": col, "values": df_first[col].tolist()})
        spark_df = pd.DataFrame(rows)

        st.dataframe(
            spark_df,
            use_container_width=True,
            column_config={
                "variable": st.column_config.TextColumn("Variable"),
                "values": st.column_config.LineChartColumn("First month", width="medium"),
            },
            hide_index=True,
        )
    else:
        st.warning("Please load weather data from the Home page first")

# --- Page 3: Plots ---
elif page == "Plots":
    st.header("Plots")
    
    if 'weather_data' in st.session_state:
        df = st.session_state.weather_data
        NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_choice = st.selectbox("Choose column", ["All (normalized)"] + NUMERIC_COLS)
        
        available_months = df["time"].dt.to_period("M").unique()
        month_options = sorted([str(month) for month in available_months])
        
        if len(month_options) > 1:
            month_range = st.select_slider("Month range", options=month_options, value=(month_options[0], month_options[0]))
        else:
            month_range = (month_options[0], month_options[0])
            st.info(f"Only one month available: {month_options[0]}")
        
        start, end = pd.Period(month_range[0], "M"), pd.Period(month_range[1], "M")
        if start > end:
            start, end = end, start
            
        mask = (df["time"].dt.to_period("M") >= start) & (df["time"].dt.to_period("M") <= end)
        dff = df.loc[mask]

        fig, ax = plt.subplots(figsize=(11, 4.5))

        if col_choice == "All (normalized)":
            for col in NUMERIC_COLS:
                mn, mx = dff[col].min(), dff[col].max()
                vals = 0.0 if mx == mn else (dff[col] - mn) / (mx - mn)
                ax.plot(dff["time"], vals, label=col)
            ax.set_ylabel("Normalized (0-1)")
        else:
            ax.plot(dff["time"], dff[col_choice], label=col_choice, color="tab:blue")
            ax.set_ylabel(col_choice)

        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        st.pyplot(fig)
    else:
        st.warning("Please load weather data from the Home page first")

# --- Page 4: Mongo Dashboard ---
elif page == "Mongo Dashboard":
    st.header("Mongo Dashboard")
    
    df_energy = get_energy_data()
    
    # Show info if using sample data
    if len(df_energy) < 100:  # Sample data is smaller
        st.info("Using sample data for demonstration")
    
    if not df_energy.empty:
        left, right = st.columns(2)

        with left:
            st.subheader("Production (Pie Chart)")
            price_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
            selected_area = st.radio("Price Area", price_areas, key="energy_dashboard")
            
            area_data = df_energy[df_energy["priceArea"] == selected_area]
            production_totals = area_data.groupby("productionGroup")["quantitykwh"].sum()

            fig1, ax1 = plt.subplots()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ax1.pie(production_totals.values, labels=production_totals.index,
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title(f"Total Production - {selected_area}")
            st.pyplot(fig1)

        with right:
            st.subheader("Monthly Production (Line Chart)")
            groups = ["hydro", "other", "solar", "thermal", "wind"]
            selected_groups = st.multiselect("Production Groups", groups, default=groups)
            
            months = ["January", "February", "March", "April", "May", "June", "July",
                     "August", "September", "October", "November", "December"]
            selected_month = st.selectbox("Months", months)
            month_number = months.index(selected_month) + 1

            if selected_groups:
                filtered_data = df_energy[
                    (df_energy["priceArea"] == selected_area) &
                    (df_energy["productionGroup"].isin(selected_groups)) &
                    (df_energy["startTime"].dt.month == month_number)
                ]
                if not filtered_data.empty:
                    pivot_df = filtered_data.pivot_table(
                        index="startTime",
                        columns="productionGroup", 
                        values="quantitykwh",
                        aggfunc="sum"
                    ).fillna(0)
                    st.line_chart(pivot_df)
                else:
                    st.info("No data available for selected filters")

# --- Page 5: STL / Spectrogram (Part 3A) ---
elif page == "STL / Spectrogram (Part 3A)":
    st.header("STL / Spectrogram Analysis (Part 3A)")
    
    df_energy = get_energy_data()
    
    # Show info if using sample data
    if len(df_energy) < 100:
        st.info("Using sample data for demonstration - STL may show limited patterns")
    
    if not df_energy.empty:
        tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])
        
        with tab1:
            st.subheader("STL Decomposition")
            col1, col2 = st.columns(2)
            
            with col1:
                price_area = st.selectbox("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"], key="stl_area")
                production_group = st.selectbox("Production Group", ["hydro", "wind", "solar", "thermal", "other"], key="stl_group")
            
            with col2:
                period = st.slider("Period", 1, 30, 7, help="Seasonal period in days")
                seasonal = st.slider("Seasonal Smoother", 5, 21, 13)
                trend = st.slider("Trend Smoother", 50, 200, 101)
            
            if st.button("Run STL Decomposition", key="stl_button"):
                with st.spinner("Performing STL decomposition..."):
                    try:
                        fig = stl_decompose_elhub(
                            df_energy, 
                            price_area=price_area,
                            production_group=production_group,
                            period=period,
                            seasonal=seasonal,
                            trend=trend
                        )
                        st.pyplot(fig)
                        st.success("STL decomposition completed!")
                    except Exception as e:
                        st.error(f"Error in STL decomposition: {e}")
        
        with tab2:
            st.subheader("Spectrogram Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                price_area = st.selectbox("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"], key="spec_area")
                production_group = st.selectbox("Production Group", ["hydro", "wind", "solar", "thermal", "other"], key="spec_group")
            
            with col2:
                window_length = st.slider("Window Length (hours)", 24, 24*30, 24*7, help="FFT window length in hours")
                overlap = st.slider("Overlap Ratio", 0.1, 0.9, 0.5, help="Window overlap ratio")
            
            if st.button("Generate Spectrogram", key="spec_button"):
                with st.spinner("Generating spectrogram..."):
                    try:
                        fig = plot_production_spectrogram(
                            df_energy,
                            price_area=price_area,
                            production_group=production_group,
                            window_length=window_length,
                            overlap=overlap
                        )
                        st.pyplot(fig)
                        st.success("Spectrogram generated!")
                    except Exception as e:
                        st.error(f"Error generating spectrogram: {e}")

# --- Page 6: Outlier / Anomaly (Part 3B) ---
elif page == "Outlier / Anomaly (Part 3B)":
    st.header("Outlier / Anomaly Detection (Part 3B)")
    
    if 'weather_data' in st.session_state:
        df = st.session_state.weather_data
        
        tab1, tab2 = st.tabs(["Temperature Outliers", "Precipitation Anomalies"])
        
        with tab1:
            st.subheader("Temperature Outlier Detection (DCT + SPC)")
            col1, col2 = st.columns(2)
            
            with col1:
                cutoff_freq = st.slider("Cutoff Frequency", 0.001, 0.1, 0.02, help="DCT high-pass cutoff frequency")
                n_sigma = st.slider("Number of Sigma", 1.0, 5.0, 3.0, help="Number of standard deviations for SPC limits")
            
            if st.button("Detect Temperature Outliers", key="temp_button"):
                with st.spinner("Detecting temperature outliers..."):
                    try:
                        fig, summary = plot_temperature_outliers(
                            df, 
                            cutoff_freq=cutoff_freq,
                            n_sigma=n_sigma
                        )
                        st.pyplot(fig)
                        
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Points", summary["n_points"])
                        col2.metric("Outliers Detected", summary["n_outliers"])
                        col3.metric("Outlier Percentage", f"{summary['outlier_percent']}%")
                        
                    except Exception as e:
                        st.error(f"Error in temperature outlier detection: {e}")
        
        with tab2:
            st.subheader("Precipitation Anomaly Detection (LOF)")
            contamination = st.slider("Contamination", 0.001, 0.1, 0.01, help="Expected proportion of anomalies")
            
            if st.button("Detect Precipitation Anomalies", key="precip_button"):
                with st.spinner("Detecting precipitation anomalies..."):
                    try:
                        fig, summary = detect_precipitation_anomalies(
                            df,
                            contamination=contamination
                        )
                        st.pyplot(fig)
                        
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Points", summary["n_points"])
                        col2.metric("Anomalies Detected", summary["n_anomalies"])
                        col3.metric("Anomaly Percentage", f"{summary['anomaly_percent']}%")
                        
                    except Exception as e:
                        st.error(f"Error in precipitation anomaly detection: {e}")
    else:
        st.warning("Please load weather data from the Home page first")

# --- Page 7: About ---
elif page == "About":
    st.header("About")
    st.markdown("""
    ### IND320 Dashboard = Parts 1, 2 & 3
    
    This application demonstrates the complete progression through all three parts of the IND320 project.
    
    **Part 1**: Basic data visualization and analysis
    - CSV data loading and processing
    - Interactive tables and plots
    - Data normalization and exploration
    
    **Part 2**: Database integration and energy analytics  
    - MongoDB connection for Elhub data
    - Interactive energy production dashboard
    - Real-time data visualization
    
    **Part 3**: Advanced time series analysis
    - Open-Meteo API integration for weather data
    - STL decomposition for trend analysis
    - Spectrogram for frequency analysis
    - Statistical outlier and anomaly detection
    
    **Technologies used**:
    - Streamlit for web interface
    - MongoDB for data storage
    - Open-Meteo API for weather data
    - Scipy, Statsmodels for analysis
    - Matplotlib for visualization
    """)