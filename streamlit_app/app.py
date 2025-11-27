from __future__ import annotations

import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from pathlib import Path 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
import json
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import timedelta
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# ---- MUST BE FIRST STREAMLIT CALL ----
st.set_page_config(page_title="IND320 - Dashboard (Part 1–4)", layout="wide")

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
def plot_temperature_outliers(
    df,
    temp_col="temperature_2m",
    time_col="time",
    cutoff_freq=0.02,
    n_sigma=3.0,
):
    """
    Detect temperature outliers using a DCT-based trend + robust SPC limits,
    and return an interactive Plotly figure + a summary dictionary.
    """
    # --- Clean and prepare data ---
    d = df[[time_col, temp_col]].copy()
    d = (
        d.dropna(subset=[temp_col])
         .drop_duplicates(subset=[time_col])
         .sort_values(time_col)
    )
    d[time_col] = pd.to_datetime(d[time_col], utc=True)
    d[temp_col] = d[temp_col].interpolate(limit_direction="both")

    # --- DCT: split into trend + residual ---
    x = d[temp_col].to_numpy()
    X = dct(x, norm="ortho")
    n = len(X)

    # cutoff_freq controls how much low-frequency content is kept in the trend
    cut_idx = int(np.clip(cutoff_freq * n, 1, n - 1))

    # 1) Low-frequency TREND: keep the first frequencies, zero out the rest
    X_trend = X.copy()
    X_trend[cut_idx:] = 0.0
    trend = idct(X_trend, norm="ortho")

    # 2) RESIDUAL: original − trend (high-frequency component)
    resid = x - trend

    # --- Robust SPC limits on the residual ---
    # We estimate a robust standard deviation using MAD
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    robust_sd = 1.4826 * mad if mad > 0 else np.std(resid)

    # SPC limits in the residual domain: median ± n_sigma * robust_sd
    upper_resid = med + n_sigma * robust_sd
    lower_resid = med - n_sigma * robust_sd

    # Time-varying limits in the temperature domain:
    # trend(t) + residual limits
    upper_series = trend + upper_resid
    lower_series = trend + lower_resid

    # Outlier if the residual is outside the SPC band
    outlier_mask = (resid > upper_resid) | (resid < lower_resid)

    # --- Plotly figure ---
    fig = go.Figure()

    # Temperature series
    fig.add_trace(
        go.Scatter(
            x=d[time_col],
            y=d[temp_col],
            mode="lines",
            name="Temperature",
        )
    )

    # Outliers as red markers
    if outlier_mask.any():
        fig.add_trace(
            go.Scatter(
                x=d.loc[outlier_mask, time_col],
                y=d.loc[outlier_mask, temp_col],
                mode="markers",
                name="Outliers",
                marker=dict(size=6, color="red"),
            )
        )

    # SPC limits (time-varying in the temperature domain)
    fig.add_trace(
        go.Scatter(
            x=d[time_col],
            y=upper_series,
            mode="lines",
            name="Upper limit",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d[time_col],
            y=lower_series,
            mode="lines",
            name="Lower limit",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="Temperature Outlier Detection – DCT trend + SPC",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    summary = {
        "n_points": int(len(d)),
        "n_outliers": int(outlier_mask.sum()),
        "outlier_percent": round(100 * outlier_mask.mean(), 2),
        "robust_sd": float(robust_sd),
    }

    return fig, summary

def detect_precipitation_anomalies(
    df,
    precip_col="precipitation",
    time_col="time",
    contamination=0.01,
):
    """
    Detect precipitation anomalies using Local Outlier Factor (LOF)
    and return an interactive Plotly figure + summary dictionary.
    """
    d = df[[time_col, precip_col]].copy()
    d = d.dropna(subset=[precip_col])
    d[time_col] = pd.to_datetime(d[time_col], utc=True)
    d = d.sort_values(time_col)

    # Time index in hours from start (for LOF)
    d["time_index"] = (d[time_col] - d[time_col].min()).dt.total_seconds() / 3600.0

    X = d[[precip_col, "time_index"]].to_numpy()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = lof.fit_predict(X)
    anomaly_mask = labels == -1

    # --- Plotly figure ---
    fig = go.Figure()

    # Main precipitation series
    fig.add_trace(
        go.Scatter(
            x=d[time_col],
            y=d[precip_col],
            mode="lines",
            name="Precipitation",
        )
    )

    # Anomalies as red markers
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=d.loc[anomaly_mask, time_col],
                y=d.loc[anomaly_mask, precip_col],
                mode="markers",
                name="Anomalies",
                marker=dict(size=6, color="red"),
            )
        )

    fig.update_layout(
        title="Precipitation Anomaly Detection – LOF",
        xaxis_title="Time",
        yaxis_title="Precipitation (mm)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    summary = {
        "n_points": int(len(d)),
        "n_anomalies": int(anomaly_mask.sum()),
        "anomaly_percent": round(100 * anomaly_mask.mean(), 2),
    }
    return fig, summary

def stl_decompose_elhub(
    df,
    price_area="NO5",
    production_group="hydro",
    period=7,
    seasonal=13,
    trend=101,
    robust=True,
):
    """
    STL decomposition of daily kWh for a price area / production group.
    Returns a Plotly figure with 4 stacked subplots:
    Original, Trend, Seasonal, Residual.
    """
    # Aggregate to daily kWh
    daily_data = (
        df[
            (df["priceArea"] == price_area)
            & (df["productionGroup"] == production_group)
        ]
        .set_index("startTime")[["quantitykwh"]]
        .resample("D")
        .sum()
        .rename(columns={"quantitykwh": "kWh"})
    )

    series = daily_data["kWh"].astype(float)

    if series.empty:
        raise ValueError("No data for selected price area / production group.")

    stl_result = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust,
    ).fit()

    # --- Build Plotly figure with 4 rows ---
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Original – {price_area}/{production_group}",
            "Trend",
            "Seasonal",
            "Residual",
        ),
    )

    dates = series.index

    # Original
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series.values,
            mode="lines",
            name="Original",
        ),
        row=1,
        col=1,
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stl_result.trend,
            mode="lines",
            name="Trend",
        ),
        row=2,
        col=1,
    )

    # Seasonal
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stl_result.seasonal,
            mode="lines",
            name="Seasonal",
        ),
        row=3,
        col=1,
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stl_result.resid,
            mode="lines",
            name="Residual",
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_yaxes(title_text="kWh", row=4, col=1)

    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig

def plot_production_spectrogram(
    df,
    price_area="NO5",
    production_group="hydro",
    window_length=24 * 7,
    overlap=0.5,
):
    """
    Generate a spectrogram of energy production using Plotly.
    Returns a Plotly figure (heatmap).
    """
    # Filter + hourly resample
    filtered = df[
        (df["priceArea"] == price_area)
        & (df["productionGroup"] == production_group)
    ].copy()

    filtered["startTime"] = pd.to_datetime(filtered["startTime"], utc=True)
    hourly_data = (
        filtered.set_index("startTime")["quantitykwh"]
        .resample("H")
        .mean()
        .interpolate()
    )

    # Compute spectrogram using SciPy
    nperseg = int(window_length)
    noverlap = int(nperseg * overlap)

    f, t, Sxx = spectrogram(
        hourly_data.values,
        fs=1 / 3600,               # 1 sample per hour
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Convert power to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-8)

    # Convert SciPy time vector to timestamps
    t_start = hourly_data.index.min()
    time_axis = t_start + pd.to_timedelta(t, unit="s")

    # Create Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=Sxx_dB,
            x=time_axis,
            y=f,
            colorscale="Viridis",
            colorbar=dict(title="Power [dB]"),
        )
    )

    fig.update_layout(
        title=f"Spectrogram – {price_area}/{production_group}",
        xaxis_title="Time",
        yaxis_title="Frequency [Hz]",
        margin=dict(l=40, r=20, t=50, b=40),
    )

    return fig

def sliding_window_correlation(
    df: pd.DataFrame,
    meteo_col: str,
    energy_col: str,
    window_hours: int = 24 * 14,
    lag_hours: int = 0,
) -> pd.DataFrame:
    """
    Compute sliding-window correlation between a meteorological and an energy series.

    Parameters
    ----------
    df : DataFrame
        Must have a datetime index and columns meteo_col and energy_col.
    meteo_col : str
        Name of meteorological column.
    energy_col : str
        Name of energy column.
    window_hours : int
        Length of the sliding window in hours.
    lag_hours : int
        Positive = energy series is shifted forward (energy responds later).
        Negative = energy leads the meteorological signal.

    Returns
    -------
    DataFrame with columns:
        - time
        - corr  (Pearson correlation in each window)
    """
    d = df[[meteo_col, energy_col]].copy().dropna().sort_index()

    if lag_hours != 0:
        d[energy_col] = d[energy_col].shift(int(lag_hours))

    win = int(window_hours)
    min_per = max(10, win // 2)  # require at least half the window

    corr = d[meteo_col].rolling(win, min_periods=min_per).corr(d[energy_col])

    out = pd.DataFrame({"time": d.index, "corr": corr})
    return out.dropna()

# --- Snow drift helper functions (Part 4B) ---
def compute_Qupot(hourly_wind_speeds, dt=3600):
    hourly_wind_speeds = np.asarray(hourly_wind_speeds, dtype=float)
    return float(np.sum((hourly_wind_speeds ** 3.8) * dt) / 233847.0)


def sector_index(direction):
    return int(((direction + 11.25) % 360) // 22.5)


def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    sectors = np.zeros(16, dtype=float)
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        sectors[sector_index(d)] += (u ** 3.8) * dt / 233847.0
    return sectors


def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe

    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    Qt = Qinf * (1 - 0.14 ** (F / T))
    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control,
    }


@st.cache_data(show_spinner=False)
def download_snowdrift_weather(lat, lon, start_year, end_year):
    url = (
        "https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_year}-07-01"
        f"&end_date={end_year + 1}-06-30"
        "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"
        "&timezone=UTC"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    years = df["time"].dt.year
    months = df["time"].dt.month
    df["season"] = np.where(months >= 7, years, years - 1)
    return df


def compute_yearly_snowdrift(df, T, F, theta, start_season, end_season):
    results = []
    for s in range(start_season, end_season + 1):
        df_s = df[df["season"] == s].copy()
        if df_s.empty:
            continue
        df_s["Swe_hourly"] = np.where(df_s["temperature_2m"] < 1.0,
                                      df_s["precipitation"], 0.0)
        Swe = df_s["Swe_hourly"].sum()
        ws = df_s["wind_speed_10m"].to_numpy()
        res = compute_snow_transport(T, F, theta, Swe, ws)
        res["season"] = s
        res["season_label"] = f"{s}-{s+1}"
        results.append(res)
    return pd.DataFrame(results)


def compute_monthly_snowdrift(df, T, F, theta, start_season, end_season):
    df = df.copy()
    df["Swe_hourly"] = np.where(df["temperature_2m"] < 1.0,
                                df["precipitation"], 0.0)

    month = df["time"].dt.month
    df["season_month"] = np.where(month >= 7, month - 6, month + 6)

    results = []
    for s in range(start_season, end_season + 1):
        df_s = df[df["season"] == s]
        if df_s.empty:
            continue
        for m in range(1, 13):
            df_sm = df_s[df_s["season_month"] == m]
            if df_sm.empty:
                continue
            Swe = df_sm["Swe_hourly"].sum()
            ws = df_sm["wind_speed_10m"].to_numpy()
            res = compute_snow_transport(T, F, theta, Swe, ws)
            res["season"] = s
            res["season_month"] = m
            results.append(res)
    return pd.DataFrame(results)


def compute_average_sector(df, start_season, end_season):
    sectors = []
    for s in range(start_season, end_season + 1):
        g = df[df["season"] == s]
        if g.empty:
            continue
        ws = g["wind_speed_10m"].to_numpy()
        wd = g["wind_direction_10m"].to_numpy()
        sectors.append(compute_sector_transport(ws, wd))
    if not sectors:
        return np.zeros(16)
    return np.mean(sectors, axis=0)


def plot_rose(avg_sector_values, overall_avg_kgm):
    """
    Plot a wind-rose style diagram using Plotly.
    
    Parameters
    ----------
    avg_sector_values : array-like of length 16
        Snow transport values per 22.5° wind sector.
    overall_avg_kgm : float
        Overall Qt in kg/m (used only for title).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive polar bar chart.
    """

    # Convert kg/m → tonnes/m (consistent with your other plots)
    vals_ton = np.array(avg_sector_values) / 1000.0

    # Directions for 16-sector rose (22.5° each)
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]

    # Angle mapping: each sector center = 0°, 22.5°, 45°, …
    angles_deg = np.arange(0, 360, 22.5)

    fig = go.Figure()

    fig.add_trace(
        go.Barpolar(
            r=vals_ton,
            theta=angles_deg,
            width=[22.5] * 16,
            marker=dict(
                line=dict(color="black", width=1)
            ),
            name="Qt (tonnes/m)"
        )
    )

    total_ton = overall_avg_kgm / 1000.0

    fig.update_layout(
        title=f"Wind Rose — Overall Qt = {total_ton:.1f} tonnes/m",
        polar=dict(
            radialaxis=dict(title="Qt (tonnes/m)", tickformat=".1f"),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,  # North at the top
                tickmode="array",
                tickvals=angles_deg,
                ticktext=directions,
            ),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )

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
    
@st.cache_resource
def get_production_data_22_24():
    """
    Load Elhub production data for 2022–2024 from MongoDB.

    Expects a collection named 'elhub_production_data_2022_2024'
    with columns:
      - pricearea
      - productiongroup
      - starttime
      - quantitykwh
    """
    try:
        # Connect to MongoDB using the same URI as elsewhere in the app
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        client.admin.command('ping')
        db = client["ind320"]

        coll = db["elhub_production_data_2022_2024"]
        data = list(coll.find({}, {"_id": 0}))
        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Normalize column names to match the rest of the app
        column_mapping = {}
        for col in df.columns:
            if col.lower() == "pricearea":
                column_mapping[col] = "priceArea"
            elif col.lower() == "productiongroup":
                column_mapping[col] = "productionGroup"
            elif col.lower() == "starttime":
                column_mapping[col] = "startTime"
            elif col.lower() in ("quantitykwh", "quantity"):
                column_mapping[col] = "quantitykwh"

        df.rename(columns=column_mapping, inplace=True)

        # Convert to proper dtypes
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")

        # Drop rows with missing core fields and sort
        df = df.dropna(subset=["priceArea", "productionGroup", "startTime", "quantitykwh"])
        df = df.sort_values("startTime")

        return df

    except Exception as e:
        st.error(f"Database error (production 22–24): {e}")
        return pd.DataFrame()


@st.cache_resource
def get_consumption_data_21_24():
    """
    Load Elhub consumption data for 2021–2024 from MongoDB.

    Expects a collection named 'elhub_consumption_data_2021_2024'
    with columns:
      - pricearea
      - consumptiongroup
      - starttime
      - quantitykwh
    """
    try:
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        client.admin.command('ping')
        db = client["ind320"]

        coll = db["elhub_consumption_data_2021_2024"]
        data = list(coll.find({}, {"_id": 0}))
        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Normalize column names
        column_mapping = {}
        for col in df.columns:
            if col.lower() == "pricearea":
                column_mapping[col] = "priceArea"
            elif col.lower() == "consumptiongroup":
                column_mapping[col] = "consumptionGroup"
            elif col.lower() == "starttime":
                column_mapping[col] = "startTime"
            elif col.lower() in ("quantitykwh", "quantity"):
                column_mapping[col] = "quantitykwh"

        df.rename(columns=column_mapping, inplace=True)

        df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")

        df = df.dropna(subset=["priceArea", "consumptionGroup", "startTime", "quantitykwh"])
        df = df.sort_values("startTime")

        return df

    except Exception as e:
        st.error(f"Database error (consumption 21–24): {e}")
        return pd.DataFrame()

# --- Sidebar Navigation ---
st.sidebar.title("IND320 Navigation")
st.sidebar.markdown("---")
sections = {
   "Overview & Data": [
       "Home",
       "Data Table",
       "Mongo Dashboard",
   ],
   "Weather & Anomalies": [
       "Plots",
       "Outlier / Anomaly (Part 3B)",
       "Meteorology & Energy",
   ],
   "Energy Structure & Forecasts": [
       "STL / Spectrogram (Part 3A)",
       "Forecasting (SARIMAX)",
   ],
   "Spatial & Snow Drift": [
       "Map & Selectors (Part 4A)",
       "Snow Drift (Part 4B)",
   ],
   "Other": [
       "About",
   ],
}
section = st.sidebar.radio("Section", list(sections.keys()), key="nav_section")
page = st.sidebar.radio("Page", sections[section], key="nav_page")
st.sidebar.markdown("---")
st.sidebar.markdown("**IND320 Dashboard – Parts 1, 2, 3 & 4**")

if page == "Home":
    st.title("IND320 - Dashboard (Parts 1–4)")
    st.markdown("Use the sidebar to navigate between pages.")
    st.markdown("---")

    st.subheader("Welcome")
    st.markdown(
        """
        This dashboard demonstrates the progression of **IND320** through Parts 1–4:

        - **Part 1** – CSV loaded locally  
          Data summary, basic tables and exploratory plots.
        - **Part 2** – MongoDB connection  
          Elhub production data in an interactive energy dashboard.
        - **Part 3** – Open-Meteo API & advanced time series  
          STL decomposition, spectrograms, and outlier/anomaly detection.
        - **Part 4** – Integrated analytics & forecasting  
          GeoJSON map of price areas, snow-drift calculations using map coordinates,  
          sliding-window correlation between meteorology and energy, and  
          SARIMAX forecasting of production/consumption with optional exogenous variables.
        """
    )
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
            month_range = st.select_slider(
                "Month range",
                options=month_options,
                value=(month_options[0], month_options[0])
            )
        else:
            month_range = (month_options[0], month_options[0])
            st.info(f"Only one month available: {month_options[0]}")
        
        start, end = pd.Period(month_range[0], "M"), pd.Period(month_range[1], "M")
        if start > end:
            start, end = end, start
            
        mask = (
            (df["time"].dt.to_period("M") >= start) &
            (df["time"].dt.to_period("M") <= end)
        )
        dff = df.loc[mask].copy()

        # ===== Plot with Plotly =====
        if col_choice == "All (normalized)":
            # Build long-format DF with normalized values for each numeric column
            norm_frames = []
            for col in NUMERIC_COLS:
                mn, mx = dff[col].min(), dff[col].max()
                if mx == mn:
                    vals = np.zeros(len(dff))
                else:
                    vals = (dff[col] - mn) / (mx - mn)
                tmp = pd.DataFrame({
                    "time": dff["time"],
                    "variable": col,
                    "value": vals,
                })
                norm_frames.append(tmp)

            plot_df = pd.concat(norm_frames, ignore_index=True)

            fig = px.line(
                plot_df,
                x="time",
                y="value",
                color="variable",
                labels={
                    "time": "Time",
                    "value": "Normalized (0–1)",
                    "variable": "Variable",
                },
                title="Normalized weather variables",
            )
        else:
            fig = px.line(
                dff,
                x="time",
                y=col_choice,
                labels={
                    "time": "Time",
                    col_choice: col_choice,
                },
                title=f"{col_choice} over time",
            )

        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

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

        # ---------- LEFT: PIE (Plotly) ----------
        with left:
            st.subheader("Production (Pie Chart)")
            price_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
            selected_area = st.radio("Price area", price_areas, key="energy_dashboard")
    
            area_data = df_energy[df_energy["priceArea"] == selected_area]
            production_totals = (
                area_data
                .groupby("productionGroup")["quantitykwh"]
                .sum()
                .reset_index()
            )

            # Plotly donut with labels outside (similar to the old chart)
            fig1 = px.pie(
                production_totals,
                values="quantitykwh",      # column with numbers
                names="productionGroup",   # column with labels (hydro, wind, etc.)
                hole=0.5                   # makes it a donut chart
            )

            fig1.update_traces(
                textposition="outside",          # put labels outside
                textinfo="percent+label",        # show percent + label
                showlegend=False                 # hide legend (optional)
            )

            fig1.update_layout(
                margin=dict(t=140, b=40, l=40, r=40)  # space around the chart
            )

            st.plotly_chart(fig1, use_container_width=True)


        # ---------- RIGHT: LINE (Plotly) ----------
        with right:
            st.subheader("Monthly production (Line Chart)")
            groups = ["hydro", "other", "solar", "thermal", "wind"]
            selected_groups = st.multiselect("Production groups", groups, default=groups)
            
            months = [
                "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"
            ]
            selected_month = st.selectbox("Month", months)
            month_number = months.index(selected_month) + 1

            if selected_groups:
                filtered_data = df_energy[
                    (df_energy["priceArea"] == selected_area) &
                    (df_energy["productionGroup"].isin(selected_groups)) &
                    (df_energy["startTime"].dt.month == month_number)
                ]
                if not filtered_data.empty:
                    pivot_df = (
                        filtered_data
                        .pivot_table(
                            index="startTime",
                            columns="productionGroup", 
                            values="quantitykwh",
                            aggfunc="sum"
                        )
                        .fillna(0)
                    )

                    # Convert to long format for Plotly
                    plot_df = (
                        pivot_df
                        .reset_index()
                        .melt(
                            id_vars="startTime",
                            var_name="productionGroup",
                            value_name="quantitykwh",
                        )
                    )

                    fig2 = px.line(
                        plot_df,
                        x="startTime",
                        y="quantitykwh",
                        color="productionGroup",
                        title=f"Production in {selected_month} – {selected_area}",
                    )
                    fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data available for the selected filters")
# --- Page X: Map & Selectors (Part 4A) ---
elif page == "Map & Selectors (Part 4A)":
    st.header("Map & Selectors – Price Areas NO1–NO5")

    # --- Load GeoJSON with Elspot areas (NO1–NO5) ---
    try:
        # Go up from streamlit_app/ → to project root → then into data/
        base_dir = Path(__file__).resolve().parent.parent
        geojson_path = base_dir / "data" / "file.geojson"

        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

    except Exception as e:
        st.error(f"Could not open GeoJSON file: {e}")
        st.stop()

    # --- Choose dataset: production or consumption ---
    data_type = st.radio("Dataset", ["Production", "Consumption"], horizontal=True)

    if data_type == "Production":
        df = get_production_data_22_24()
        group_col = "productionGroup"
        group_label = "Production group"
    else:
        df = get_consumption_data_21_24()
        group_col = "consumptionGroup"
        group_label = "Consumption group"

    if df.empty:
        st.warning("No data loaded from MongoDB for the selected dataset.")
        st.stop()

    # --- Select group (hydro, wind, cabin, industry, etc.) ---
    groups = sorted(df[group_col].dropna().unique().tolist())
    selected_group = st.selectbox(group_label, groups)

    # --- Select time interval ---
    min_date = df["startTime"].min().date()
    max_date = df["startTime"].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
    with col2:
        n_days = st.number_input(
            "Interval length (days)",
            min_value=1,
            max_value=365,
            value=7,
            step=1,
        )

    end_date = start_date + timedelta(days=int(n_days))

    # Filter data using only the date part to avoid timezone issues
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    mask = (
        (df[group_col] == selected_group)
        & (df["startTime"].dt.date >= start_date)
        & (df["startTime"].dt.date < end_date)
    )
    dff = df.loc[mask]


    if dff.empty:
        st.info("No rows in selected date interval.")
        st.stop()

    # --- Compute mean kWh per price area (NO1–NO5) ---
    area_mean = (
        dff.groupby("priceArea")["quantitykwh"]
        .mean()
        .reset_index()
        .rename(columns={"quantitykwh": "mean_kwh"})
    )

    st.subheader("Mean kWh per price area")
    st.dataframe(area_mean)

    mean_dict = dict(zip(area_mean["priceArea"], area_mean["mean_kwh"]))
    vmin = min(mean_dict.values())
    vmax = max(mean_dict.values())

    # --- Choose price area to highlight ---
    highlight_area = st.selectbox(
        "Highlight price area",
        ["None"] + sorted(mean_dict.keys()),
    )

    # --- Helper: detect NO1–NO5 inside GeoJSON properties ---
    PRICE_CODES = {"NO1", "NO2", "NO3", "NO4", "NO5"}

    def get_feature_price_area(feature):
        props = feature.get("properties", {})
        for val in props.values():
            if isinstance(val, str) and val.upper() in PRICE_CODES:
                return val.upper()
        return None

    # --- Create Folium map centered on Norway ---
    m = folium.Map(location=[65, 13], zoom_start=4)

    def style_function(feature):
        pa = get_feature_price_area(feature)

        # base style
        style = {
            "weight": 1.0,
            "color": "black",
            "fillOpacity": 0.6,
        }

        # color by mean kWh
        if pa in mean_dict and vmax > vmin:
            r = (mean_dict[pa] - vmin) / (vmax - vmin + 1e-9)
            red = int(255 * r)
            style["fillColor"] = f"#{red:02x}0000"
        else:
            style["fillColor"] = "#cccccc"

        # highlight selected PA
        if highlight_area != "None" and pa == highlight_area:
            style["color"] = "blue"
            style["weight"] = 3.0

        return style

    gj = folium.GeoJson(
        geojson_data,
        name="Elspot Areas",
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=list(geojson_data["features"][0]["properties"].keys()),
            sticky=True,
        ),
    )
    gj.add_to(m)

    # Show clicked coordinates
    clicked = st.session_state.get("clicked_coord")
    if clicked:
        folium.Marker(clicked, popup="Selected point").add_to(m)

    # Enable click in map
    folium.LatLngPopup().add_to(m)

    # Render map
    map_state = st_folium(m, width=900, height=600)

    # Store click into session_state
    if map_state and map_state.get("last_clicked"):
        lat = map_state["last_clicked"]["lat"]
        lon = map_state["last_clicked"]["lng"]
        st.session_state["clicked_coord"] = (lat, lon)
        st.info(f"Saved coordinate: lat={lat:.4f}, lon={lon:.4f}")

    st.caption(
        "This coordinate will be used later in the Snow Drift (Part 4B) page."
    )
# --- Page: Snow Drift (Part 4B) ---
elif page == "Snow Drift (Part 4B)":
    st.header("Snow Drift Calculation and Illustration")

    coord = st.session_state.get("clicked_coord")
    if coord is None:
        st.warning("No coordinate selected. Go to the map page and click a location first.")
        st.stop()

    lat, lon = coord
    st.info(f"Using coordinate: lat={lat:.4f}, lon={lon:.4f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Transport distance T (m)", 100.0, value=3000.0)
    with col2:
        F = st.number_input("Fetch distance F (m)", 1000.0, value=30000.0)
    with col3:
        theta = st.number_input("Relocation coefficient θ", 0.0, 1.0, value=0.5)

    y1, y2 = st.columns(2)
    with y1:
        start_year = st.number_input("Start season year", 2000, 2025, 2015)
    with y2:
        end_year = st.number_input("End season year", 2000, 2025, 2020)

    if st.button("Compute snow drift"):
        # 1)  Check if the year start to end is right
        if start_year > end_year:
            st.error("Start season year must be less than or equal to end season year.")
        else:
            with st.spinner("Downloading weather and computing drift..."):
                df = download_snowdrift_weather(lat, lon, int(start_year), int(end_year))
                yearly = compute_yearly_snowdrift(df, T, F, theta, int(start_year), int(end_year))
                monthly = compute_monthly_snowdrift(df, T, F, theta, int(start_year), int(end_year))

                if yearly.empty:
                    st.error("No yearly results computed.")
                    st.stop()

                yearly["Qt_ton"] = yearly["Qt (kg/m)"] / 1000
                overall_avg = yearly["Qt (kg/m)"].mean()

                st.subheader("Yearly snow drift (Qt)")
                st.dataframe(yearly[["season_label", "Qt_ton", "Control"]])

                st.subheader("Yearly and Monthly Snow Drift")

                # 2) Plot yearly and monthly Qt using Plotly
                if monthly.empty:
                    st.warning("No monthly results computed.")
                else:
                    monthly["Qt_ton"] = monthly["Qt (kg/m)"] / 1000
                    monthly_mean = monthly.groupby("season_month")["Qt_ton"].mean()
                    month_labels = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                                    "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

                    # Create two stacked Plotly subplots
                    fig_qt = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.15,
                        subplot_titles=(
                            "Yearly Qt (tonnes/m)",
                            "Average Monthly Qt (tonnes/m)",
                        ),
                    )

                    # Row 1: yearly Qt
                    fig_qt.add_trace(
                        go.Scatter(
                            x=yearly["season_label"],
                            y=yearly["Qt_ton"],
                            mode="lines+markers",
                            name="Yearly Qt",
                        ),
                        row=1,
                        col=1,
                    )

                    # Row 2: average monthly Qt
                    fig_qt.add_trace(
                        go.Scatter(
                            x=month_labels,
                            y=monthly_mean.values,
                            mode="lines+markers",
                            name="Average monthly Qt",
                        ),
                        row=2,
                        col=1,
                    )

                    fig_qt.update_xaxes(title_text="Season", row=1, col=1)
                    fig_qt.update_xaxes(title_text="Season month", row=2, col=1)

                    fig_qt.update_yaxes(title_text="Qt (tonnes/m)", row=1, col=1)
                    fig_qt.update_yaxes(title_text="Qt (tonnes/m)", row=2, col=1)

                    fig_qt.update_layout(
                        height=600,
                        margin=dict(l=40, r=20, t=80, b=40),
                        showlegend=False,
                    )

                    st.plotly_chart(fig_qt, use_container_width=True)
                # Wind rose
                st.subheader("Wind Rose")
                avg_sector = compute_average_sector(df, int(start_year), int(end_year))
                fig_rose = plot_rose(avg_sector, overall_avg)
                st.plotly_chart(fig_rose, use_container_width=True)
# --- Page: Meteorology & Energy ---
# --- Page: Meteorology & Energy ---
elif page == "Meteorology & Energy":
    st.header("Meteorology and Energy – Sliding Window Correlation")

    # 1) Select price area
    area = st.selectbox("Price area", ["NO1", "NO2", "NO3", "NO4", "NO5"])

    # 2) Select meteorological variable
    meteo_options = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Relative humidity (%)": "relative_humidity_2m",
        "Wind speed (m/s)": "wind_speed_10m",
        "Pressure (hPa)": "pressure_msl",
    }
    meteo_label = st.selectbox("Meteorological variable", list(meteo_options.keys()))
    meteo_col = meteo_options[meteo_label]

    # 3) Select energy dataset (production / consumption)
    energy_mode = st.radio(
        "Energy dataset",
        ["Production (2021)", "Consumption (2021)"],
        horizontal=True,
    )

    # 4) Select lag and window length
    lag_hours = st.slider(
        "Lag (hours, positive = energy responds later)",
        min_value=-72,
        max_value=72,
        value=0,
        step=3,
    )
    window_days = st.slider(
        "Window length (days)",
        min_value=3,
        max_value=60,
        value=14,
        step=1,
    )
    window_hours = window_days * 24

    if st.button("Compute correlation", type="primary"):
        with st.spinner("Downloading weather data and computing correlation..."):

            # --- Weather data for selected area and year 2021 ---
            city = cities[area]
            weather_df = download_weather_data(city["lat"], city["lon"], year=2021)
            weather_df = (
                weather_df
                .set_index("time")
                .sort_index()
                .resample("H")
                .mean()
            )

            # --- Energy data ---
            if energy_mode.startswith("Production"):
                df_energy_all = get_energy_data()  # 2021 production
                df_energy_sel = df_energy_all[
                    (df_energy_all["priceArea"] == area)
                    & (df_energy_all["startTime"].dt.year == 2021)
                ]
                energy_label = "Production (kWh)"
            else:
                df_cons_all = get_consumption_data_21_24()
                df_energy_sel = df_cons_all[
                    (df_cons_all["priceArea"] == area)
                    & (df_cons_all["startTime"].dt.year == 2021)
                ]
                energy_label = "Consumption (kWh)"

            if df_energy_sel.empty:
                st.error("No energy data found for this area/year.")
                st.stop()

            energy_series = (
                df_energy_sel
                .set_index("startTime")["quantitykwh"]
                .sort_index()
                .resample("H")
                .sum()
            )

            # Make index timezone-naive to match weather_df (which has tz-naive timestamps)
            if getattr(energy_series.index, "tz", None) is not None:
                energy_series.index = energy_series.index.tz_localize(None)

            # --- Merge meteo and energy ---
            df_join = pd.concat(
                [
                    weather_df[meteo_col].rename("meteo"),
                    energy_series.rename("energy"),
                ],
                axis=1,
            ).dropna()

            if df_join.empty:
                st.error("No overlapping meteorological/energy data in 2021 for this area.")
                st.stop()

            # --- Normalize for time-series plot (z-score) ---
            df_plot = df_join.copy()
            df_plot["meteo_norm"] = (
                df_plot["meteo"] - df_plot["meteo"].mean()
            ) / df_plot["meteo"].std(ddof=0)
            df_plot["energy_norm"] = (
                df_plot["energy"] - df_plot["energy"].mean()
            ) / df_plot["energy"].std(ddof=0)
            df_plot = df_plot.reset_index().rename(columns={"index": "time"})

            # --- Sliding window correlation ---
            df_corr = sliding_window_correlation(
                df_join,
                meteo_col="meteo",
                energy_col="energy",
                window_hours=window_hours,
                lag_hours=lag_hours,
            )

        # ===== Plot 1: time series (normalized) =====
        st.subheader("Time series (normalized)")

        fig_ts = px.line(
            df_plot,
            x="time",
            y=["meteo_norm", "energy_norm"],
            labels={
                "value": "Normalized value",
                "time": "Time",
                "variable": "Series"
            },
        )
        fig_ts.update_layout(
            legend_title_text="Series",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ===== Plot 2: sliding window correlation =====
        st.subheader("Sliding window correlation")

        if df_corr.empty:
            st.warning("Correlation series is empty (too short or too many NaNs).")
        else:
            fig_corr = px.line(
                df_corr,
                x="time",
                y="corr",
                labels={"corr": "Correlation", "time": "Time"},
            )
            fig_corr.add_hline(y=0, line_dash="dash")
            fig_corr.update_yaxes(range=[-1.05, 1.05])
            fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            "Tip: Try different window lengths and lags to look for changes in "
            "correlation during normal conditions and during/after extreme weather events."
        )
elif page == "Forecasting (SARIMAX)":
    st.header("Forecasting Energy Production & Consumption (SARIMAX)")

    # Select production or consumption dataset
    dataset_option = st.radio(
        "Dataset",
        ["Production (2022–2024)", "Consumption (2021–2024)"],
        horizontal=True,
    )

    if dataset_option.startswith("Production"):
        df_all = get_production_data_22_24()
        group_col = "productionGroup"
        data_label = "Production (kWh)"
    else:
        df_all = get_consumption_data_21_24()
        group_col = "consumptionGroup"
        data_label = "Consumption (kWh)"

    if df_all.empty:
        st.error("No data loaded from MongoDB.")
        st.stop()

    # Select price area and group
    area = st.selectbox("Price Area", sorted(df_all["priceArea"].unique()))
    group_list = sorted(df_all.loc[df_all["priceArea"] == area, group_col].unique())
    target_group = st.selectbox("Target Group", group_list)

    # Extract target time series and resample to daily values
    df_target = df_all[
        (df_all["priceArea"] == area)
        & (df_all[group_col] == target_group)
    ].copy()

    df_target["startTime"] = pd.to_datetime(df_target["startTime"])
    ts_y = (
        df_target.set_index("startTime")["quantitykwh"]
        .sort_index()
        .resample("D")
        .sum()
    )

    # Make index timezone-naive (remove any timezone information)
    ts_y.index = ts_y.index.tz_localize(None)

    if ts_y.empty:
        st.error("No time series data available for this selection.")
        st.stop()

    # Exogenous variables: other groups
    possible_exog = [g for g in group_list if g != target_group]
    exog_selection = st.multiselect("Exogenous variables", possible_exog)

    if exog_selection:
        exog_series_list = []
        for g in exog_selection:
            df_g = df_all[
                (df_all["priceArea"] == area)
                & (df_all[group_col] == g)
            ].copy()
            df_g["startTime"] = pd.to_datetime(df_g["startTime"])

            daily = (
                df_g.set_index("startTime")["quantitykwh"]
                .sort_index()
                .resample("D")
                .sum()
                .rename(g)
            )

            # Make the exogenous series tz-naive (always safe)
            daily.index = daily.index.tz_localize(None)

            exog_series_list.append(daily)

        # Align all exogenous series with the target index
        exog_df = pd.concat(exog_series_list, axis=1)
        exog_df = exog_df.reindex(ts_y.index).fillna(0.0)
    else:
        exog_df = None

    # --- Training period & forecast horizon ---
    st.subheader("Training Period and Forecast Horizon")

    min_date = ts_y.index.min().date()
    max_date = ts_y.index.max().date()

    colA, colB, colC = st.columns(3)
    with colA:
        train_start = st.date_input(
            "Training start",
            min_date,
            min_value=min_date,
            max_value=max_date,
        )
    with colB:
        # default: end 30 days before last date
        default_end = max_date - timedelta(days=30)
        if default_end < min_date:
            default_end = max_date
        train_end = st.date_input(
            "Training end",
            default_end,
            min_value=min_date,
            max_value=max_date,
        )
    with colC:
        horizon_days = st.number_input(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=120,
            value=30,
        )

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)

    if train_start >= train_end:
        st.warning("Training start must be earlier than training end.")
        st.stop()

    forecast_end = train_end + pd.Timedelta(days=horizon_days)

    # Comparison is safe (everything tz-naive)
    if forecast_end > ts_y.index.max():
        st.warning(
            "Forecast horizon extends beyond available data. "
            "Reduce the horizon or move the training end date backwards."
        )
        st.stop()

    # Train/test split
    y_train = ts_y.loc[train_start:train_end]
    y_test = ts_y.loc[train_end + pd.Timedelta(days=1):forecast_end]

    if exog_df is not None:
        exog_train = exog_df.loc[y_train.index]
        exog_test = exog_df.loc[y_test.index]
    else:
        exog_train = exog_test = None

    if len(y_train) < 10:
        st.error("Training period is too short. Increase the length of the training window.")
        st.stop()

    # --- SARIMAX Parameters ---
    st.subheader("SARIMAX Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        p = st.number_input("p", 0, 5, 1)
        q = st.number_input("q", 0, 5, 1)
    with col2:
        d = st.number_input("d", 0, 2, 0)
        D = st.number_input("D", 0, 2, 0)
    with col3:
        P = st.number_input("P", 0, 5, 0)
        Q = st.number_input("Q", 0, 5, 0)
        s = st.number_input("Seasonal period (s)", 1, 365, 7)

    run_forecast = st.button("Run SARIMAX Forecast")

    if run_forecast:
        with st.spinner("Fitting SARIMAX model..."):
            try:
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)

                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)

                forecast_obj = res.get_forecast(
                    steps=len(y_test),
                    exog=exog_test
                )

                mean_fc = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int()

                fc_df = pd.DataFrame({
                    "date": y_test.index,
                    "forecast": mean_fc.values,
                    "lower": conf_int.iloc[:, 0].values,
                    "upper": conf_int.iloc[:, 1].values,
                    "actual": y_test.values,
                })

            except Exception as e:
                st.error(f"SARIMAX error: {e}")
                st.stop()

        # --- Plot ---
        st.subheader("Forecast Result")

        fig = go.Figure()

        # Observed full series
        fig.add_trace(go.Scatter(
            x=ts_y.index, y=ts_y.values,
            mode="lines", name="Observed"
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["forecast"],
            mode="lines", name="Forecast"
        ))

        # Confidence interval band
        fig.add_trace(go.Scatter(
            x=list(fc_df["date"]) + list(fc_df["date"])[::-1],
            y=list(fc_df["upper"]) + list(fc_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(0, 120, 180, 0.2)",
            line=dict(width=0),
            name="95% CI"
        ))

        # Actual test values
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["actual"],
            mode="lines", name="Actual (test)",
            line=dict(dash="dot")
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{data_label} (daily)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Dynamic SARIMAX forecast using the selected training period, target variable "
            "and optional exogenous predictors. Confidence intervals show uncertainty."
        )
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
                        st.plotly_chart(fig, use_container_width=True)
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
                        st.plotly_chart(fig, use_container_width=True)
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
                cutoff_freq = st.slider("Cutoff Frequency", 0.001, 0.1, 0.02, 
                                         help="DCT high-pass cutoff frequency"
                )
                n_sigma = st.slider("Number of Sigma", 1.0, 5.0, 3.0, help="Number of standard deviations for SPC limits")
            
            if st.button("Detect Temperature Outliers", key="temp_button"):
                with st.spinner("Detecting temperature outliers..."):
                    try:
                        fig, summary = plot_temperature_outliers(
                            df, 
                            cutoff_freq=cutoff_freq,
                            n_sigma=n_sigma
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                        st.plotly_chart(fig, use_container_width=True)
                        
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
    ### IND320 Dashboard – Parts 1, 2, 3 & 4
    
    This application demonstrates the complete progression through all four parts of the IND320 project.
    
    **Part 1**: Basic data visualization and analysis  
    - CSV data loading and processing  
    - Interactive tables and plots  
    - Data normalization and exploration  
    
    **Part 2**: Database integration and energy analytics  
    - MongoDB connection for Elhub data  
    - Interactive energy production dashboard  
    
    **Part 3**: Advanced time series analysis  
    - Open-Meteo API integration for weather data  
    - STL decomposition for trend analysis  
    - Spectrogram for frequency analysis  
    - Statistical outlier and anomaly detection  
    
    **Part 4**: Spatial analysis, snow drift and forecasting  
    - GeoJSON-based map of Norwegian price areas (NO1–NO5)  
    - Snow drift calculations and wind rose based on map coordinates  
    - Sliding-window correlation between meteorology and energy  
    - SARIMAX forecasting with selectable parameters and exogenous variables  
    
    **Technologies used**:
    - Streamlit for the web interface  
    - MongoDB & Cassandra for data storage  
    - Open-Meteo and Elhub APIs for external data  
    - NumPy, Pandas, SciPy, Statsmodels, scikit-learn for analysis  
    - Matplotlib, Plotly and Folium for visualization  
    """)