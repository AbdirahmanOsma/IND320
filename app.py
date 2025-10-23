from __future__ import annotations


import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


st.set_page_config(page_title="IND320 - Part 2", layout="wide")
st.sidebar.title("IND320 Navigation")


# --- Project 1 Data ---
DATA_FILE = Path(__file__).resolve().parent / "data" / "open-meteo-subset.csv"

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV, converts 'time' to datetime and sorts"""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df 

try:
    df = load_data(DATA_FILE)
    NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
    MONTHS = (
        pd.period_range(df["time"].min().to_period("M"),
                        df["time"].max().to_period("M"),
                        freq="M")
        .astype(str)
        .tolist()
    )
    project1_loaded = True
except:
    project1_loaded = False                

    
# --- Page Navigation ---
pages = ["Home", "Project 1: Table", "Project 1: Plots", "Project 2: Energy Dashboard", "About"]
page = st.sidebar.radio("Navigate", pages)

# --- Page 1:  Home ---
if page == "Home":
    st.title("IND320 - Part 2 Dashboard")
    
    if project1_loaded:
        st.dataframe(df.head())
    else:
        st.warning("Project 1 data not available")
    

# --- Page 2: Project 1 Table --- 
elif page == "Project 1: Table" and project1_loaded:
    st.header("Project 1: Table")

    first_month = pd.Period(MONTHS[0], "M")
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
        hide_index =True,
    )

# --- Page 3: Project 1 Plots ---
elif page == "Project 1: Plots" and project1_loaded:
    st.header("Project 1: Plots")


    col_choice = st.selectbox("Choose column", ["All (normalized)"] + NUMERIC_COLS)
    month_range = st.select_slider("Month range", options=MONTHS, value=(MONTHS[0], MONTHS[0]))
    
    start, end = pd.Period(month_range[0], "M"), pd.Period(month_range[1], "M")
    if start > end:
        start, end = end, start
        
    mask = (df["time"].dt.to_period("M") >= start) & (df["time"].dt.to_period("M") <= end)
    dff = df.loc[mask]

    fig,ax = plt.subplots(figsize=(11, 4.5))


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

# --- Page 4: Project 2 Energy Dashboard --- 
elif page == "Project 2: Energy Dashboard":
    st.header("Energy Production Dashboard")

    try:
        # MongoDB connection
        client = MongoClient(
            "mongodb+srv://ind320_user:Calanka12@ind320.762ezjs.mongodb.net/?retryWrites=true&w=majority&appName=ind320",
            server_api=ServerApi('1')
        )
        client.admin.command('ping')

        # Load the data 
        database = client["ind320"]
        collection = database["elhub_production_data_2021"]
        data = list(collection.find({}, {"_id": 0}))
        df_energy = pd.DataFrame(data)
        df_energy["starttime"] = pd.to_datetime(df_energy["starttime"])

        

        # Two columns
        left, right = st.columns(2)

        # Left: Pie Chart
        with left:
            st.subheader("Production (Pie Chart)")

            # Price Area radio buttons
            price_areas =["NO1", "NO2", "NO3", "NO4", "NO5"]
            selected_area = st.radio("Price Area", price_areas)

            # Pie chart
            area_data = df_energy[df_energy["pricearea"] == selected_area]
            production_totals = area_data.groupby("productiongroup")["quantitykwh"].sum()

            

            fig1, ax1 = plt.subplots()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            ax1.pie(
                production_totals.values,
                labels=production_totals.index,
                autopct = '%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title(f"Total Production - {selected_area}")
            st.pyplot(fig1)

        # Right: Line chart
        with right:
            st.subheader("Monthly Production (Line Chart)")

            # Production Group pills
            groups = ["hydro", "other", "solar", "thermal", "wind"]
            selected_groups = st.multiselect("Production Groups", groups, default=groups)

            # Month selection
            months = ["January", "February", "March", "April", "May", "June", "July",
                      "August", "September", "October", "November", "December"]
            selected_month = st.selectbox("Months", months)
            month_number = months.index(selected_month) + 1

            if selected_groups:
                # We will filter and plot
                filtered_data = df_energy[
                    (df_energy["pricearea"] == selected_area) &
                    (df_energy["productiongroup"].isin(selected_groups))&
                    (df_energy["starttime"].dt.month == month_number)
                ]
                if not filtered_data.empty:
                    pivot_df = filtered_data.pivot_table(
                        index="starttime",
                        columns="productiongroup",
                        values="quantitykwh", 
                        aggfunc="sum"
                    ).fillna(0)
                
                    st.line_chart(pivot_df)
                else:
                    st.info("No data available for selected filters")

    except Exception as e:
        st.error(f"Database error: {e}")

    # Data source expander
    with st.expander("Data Source Information"):
        st.write("Source: Elhub API")
        st.write("Processing: Spark --> Cassandra --> MongoDB")
        st.write("Year: 2021")

# --- Page 5: About ---
elif page == "About":
    st.header("About")
    st.write("IND320 Project - Complete Dashboard 1 and 2")

# Error handling for missing Project 1 Data
elif not project1_loaded and page in ["Project 1: Table", "Project 1: Plots"]:
    st.error("Project 1 data file not found")

