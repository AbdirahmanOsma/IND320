from __future__ import annotations
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

st.set_page_config(page_title="IND320 Dashboard", layout="wide")
st.sidebar.title("Navigation")

# --- Project 1 Data ---
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "open-meteo-subset.csv"

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)

try:
    df = load_data(DATA_FILE)
    NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
    MONTHS = pd.period_range(df["time"].min().to_period("M"),
                             df["time"].max().to_period("M"),
                             freq="M").astype(str).tolist()
    project1_loaded = True
except Exception:
    project1_loaded = False

pages = ["Home", "Project 1: Table", "Project 1: Plots", "Project 2: Energy Dashboard", "About"]
page = st.sidebar.radio("Select Page", pages)

# --- Home ---
if page == "Home":
    st.title("IND320 Dashboard")
    if project1_loaded:
        st.dataframe(df.head())
    else:
        st.warning("Project 1 data not available")

# --- Project 1 Table ---
elif page == "Project 1: Table" and project1_loaded:
    st.header("Project 1 – Table View")
    first_month = pd.Period(MONTHS[0], "M")
    dff = df.loc[df["time"].dt.to_period("M") == first_month, ["time"] + NUMERIC_COLS]

    data_rows = [{"variable": col, "values": dff[col].tolist()} for col in NUMERIC_COLS]
    st.dataframe(pd.DataFrame(data_rows), use_container_width=True)

# --- Project 1 Plots ---
elif page == "Project 1: Plots" and project1_loaded:
    st.header("Project 1 – Plots")
    col_choice = st.selectbox("Variable", ["All (normalized)"] + NUMERIC_COLS)
    month_range = st.select_slider("Month range", options=MONTHS, value=(MONTHS[0], MONTHS[0]))
    start, end = sorted([pd.Period(m, "M") for m in month_range])

    dff = df.loc[(df["time"].dt.to_period("M") >= start) & (df["time"].dt.to_period("M") <= end)]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    if col_choice == "All (normalized)":
        for col in NUMERIC_COLS:
            series = dff[col]
            norm = (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else series
            ax.plot(dff["time"], norm, label=col)
        ax.set_ylabel("Normalized (0–1)")
    else:
        ax.plot(dff["time"], dff[col_choice], label=col_choice)
        ax.set_ylabel(col_choice)

    ax.set_xlabel("Time")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# --- Project 2: MongoDB Energy Dashboard ---
elif page == "Project 2: Energy Dashboard":
    st.header("Energy Production Dashboard")
    try:
        client = MongoClient(st.secrets["MONGO_URI"], server_api=ServerApi('1'))
        client.admin.command("ping")

        col = client["ind320"]["elhub_production_data_2021"]
        df_energy = pd.DataFrame(col.find({}, {"_id": 0}))
        df_energy["starttime"] = pd.to_datetime(df_energy["starttime"])

        left, right = st.columns(2)

        with left:
            st.subheader("Production by Group (Pie Chart)")
            area = st.radio("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"])
            d = df_energy[df_energy["pricearea"] == area]
            totals = d.groupby("productiongroup")["quantitykwh"].sum()

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
                pivot = d.pivot_table(index="starttime", columns="productiongroup", values="quantitykwh", aggfunc="sum")
                st.line_chart(pivot)
            else:
                st.info("No data available for this selection.")
    except Exception as e:
        st.error(f"Database error: {e}")

# --- About ---
elif page == "About":
    st.header("About")
    st.write("IND320 – Project Dashboard (Part 1 & 2)")
