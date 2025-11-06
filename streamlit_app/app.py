from __future__ import annotations
import os
from dotenv import load_dotenv
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load local .env if present (handy for localhost)
load_dotenv()

st.set_page_config(page_title="IND320 Dashboard", layout="wide")
st.sidebar.title("Navigation")

# ---------- Project 1 Data ----------
# If your CSV lives at repo root /data, and this file is in /streamlit_app:
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "open-meteo-subset.csv"

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)

try:
    df = load_data(DATA_FILE)
    NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
    MONTHS = pd.period_range(
        df["time"].min().to_period("M"),
        df["time"].max().to_period("M"),
        freq="M"
    ).astype(str).tolist()
    project1_loaded = True
except Exception as e:
    st.warning(f"Project 1 data not available: {e}")
    project1_loaded = False

pages = ["Home", "Project 1: Table", "Project 1: Plots", "Project 2: Energy Dashboard", "About"]
page = st.sidebar.radio("Select Page", pages)

# ---------- Home ----------
if page == "Home":
    st.title("IND320 Dashboard")
    if project1_loaded:
        st.dataframe(df.head())
    else:
        st.info("Project 1 data file not found.")

# ---------- Project 1: Table ----------
elif page == "Project 1: Table" and project1_loaded:
    st.header("Project 1 – Table View")
    first_month = pd.Period(MONTHS[0], "M")
    dff = df.loc[df["time"].dt.to_period("M") == first_month, ["time"] + NUMERIC_COLS]
    rows = [{"variable": col, "values": dff[col].tolist()} for col in NUMERIC_COLS]
    
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        column_config={
            "variable": st.column_config.TextColumn("Variable"),
            "values": st.column_config.LineChartColumn("First month", width="medium"),
        },
        hide_index=True,
    )

# ---------- Project 1: Plots ----------
elif page == "Project 1: Plots" and project1_loaded:
    st.header("Project 1 – Plots")
    col_choice = st.selectbox("Variable", ["All (normalized)"] + NUMERIC_COLS)
    month_range = st.select_slider("Month range", options=MONTHS, value=(MONTHS[0], MONTHS[0]))
    start, end = sorted([pd.Period(m, "M") for m in month_range])

    dff = df.loc[(df["time"].dt.to_period("M") >= start) & (df["time"].dt.to_period("M") <= end)]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    if col_choice == "All (normalized)":
        for col in NUMERIC_COLS:
            s = dff[col]
            norm = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s
            ax.plot(dff["time"], norm, label=col)
        ax.set_ylabel("Normalized (0–1)")
    else:
        ax.plot(dff["time"], dff[col_choice], label=col_choice)
        ax.set_ylabel(col_choice)

    ax.set_xlabel("Time")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ---------- Project 2: Energy Dashboard (MongoDB) ----------
elif page == "Project 2: Energy Dashboard":
    st.header("Energy Production Dashboard")

    try:
        # FIXED: Use st.secrets for Streamlit Cloud, fallback to .env for local
        if hasattr(st, 'secrets') and 'MONGO_URI' in st.secrets:
            mongodb_uri = st.secrets['MONGO_URI']
        else:
            mongodb_uri = os.getenv("MONGODB_URI")
        
        if not mongodb_uri:
            st.error("No MongoDB URI found. Set MONGO_URI in Streamlit secrets or MONGODB_URI in environment.")
            st.stop()

        client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
        client.admin.command("ping")

        col = client["ind320"]["elhub_production_data_2021"]
        df_energy = pd.DataFrame(col.find({}, {"_id": 0}))
        if df_energy.empty:
            st.info("No documents found in MongoDB collection.")
        else:
            df_energy["starttime"] = pd.to_datetime(df_energy["starttime"])

            left, right = st.columns(2)

            with left:
                st.subheader("Production by Group (Pie Chart)")
                area = st.radio("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"])
                d = df_energy[df_energy["pricearea"] == area]
                totals = d.groupby("productiongroup")["quantitykwh"].sum().sort_values(ascending=False)

                if totals.empty:
                    st.info("No data for the selected area.")
                else:
                    fig1, ax1 = plt.subplots()
                    ax1.pie(totals.values, labels=totals.index, autopct="%1.1f%%", startangle=90)
                    ax1.set_title(f"Total Production – {area}")
                    st.pyplot(fig1)

            with right:
                st.subheader("Monthly Production (Line Chart)")
                groups = ["hydro", "other", "solar", "thermal", "wind"]
                selected = st.multiselect("Groups", groups, default=groups)
                month = st.selectbox("Month", list(range(1, 13)))

                d = df_energy[
                    (df_energy["pricearea"] == area) &
                    (df_energy["productiongroup"].isin(selected)) &
                    (df_energy["starttime"].dt.month == month)
                ]

                if not d.empty:
                    pivot = d.pivot_table(
                        index="starttime", columns="productiongroup",
                        values="quantitykwh", aggfunc="sum"
                    ).fillna(0)
                    st.line_chart(pivot)
                else:
                    st.info("No data available for this selection.")
    except Exception as e:
        st.error(f"Database error: {e}")

# ---------- About ----------
elif page == "About":
    st.header("About")
    st.write("IND320 – Project Dashboard (Part 1 & 2)")