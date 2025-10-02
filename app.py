from __future__ import annotations


import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 


st.set_page_config(page_title="IND320 - Part 1", layout="wide")
st.sidebar.title("IND320 - Part 1")


DATA_FILE = Path(__file__).resolve().parent / "data" / "open-meteo-subset.csv"

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV, converts 'time' to datetime and sorts"""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df 

df = load_data(DATA_FILE)
NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
MONTHS = (
    pd.period_range(df["time"].min().to_period("M"),
                    df["time"].max().to_period("M"),
                    freq="M")
    .astype(str)
    .tolist()
                    
)

page = st.sidebar.radio("Navigate", ["Home", "Table", "Plots", "Notes"])

# 1) Home
if page == "Home":
    st.title("IND320 - Part 1 Dashboard")
    st.subheader("Preview of data")
    st.dataframe(df.head())


# 2) Table
elif page == "Table":
    st.header("Table")
    # First month
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

# --- 3) Plots ---
elif page == "Plots":
    st.header("Plots")
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
else:
    st.empty()


