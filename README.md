# IND320
IND320 – Energy Data Engineering & Analytics Project

Overview

This project retrieves, processes, stores, and visualizes Norwegian electricity production and consumption data using the Elhub API. The system integrates data engineering, database storage, and interactive analytics in a complete pipeline.

The application demonstrates skills in:
	•	API data extraction
	•	Data preprocessing
	•	Distributed processing
	•	Database integration
	•	Interactive dashboards
	•	Anomaly detection

⸻

Data Sources

Hourly energy data is retrieved from the Elhub API:
	•	Production data (2022–2024)
	•	Consumption data (2021–2024)
	•	All Norwegian price areas (NO1–NO5)

⸻

Tech Stack

Languages
	•	Python

Libraries
	•	Pandas
	•	PySpark
	•	Scikit-learn
	•	Plotly
	•	Streamlit

Databases
	•	Cassandra (distributed storage)
	•	MongoDB Atlas (document storage)

⸻

System Architecture

Pipeline workflow:
	1.	Fetch hourly data from API
	2.	Normalize and clean datasets
	3.	Store structured data
	•	Cassandra → analytics queries
	•	MongoDB → document storage
	4.	Load data into Streamlit dashboard
	5.	Visualize trends and detect anomalies

⸻

Features
	•	Interactive dashboard
	•	Multi-year energy analysis
	•	Price area comparison
	•	Outlier detection using LOF
	•	Choropleth map visualization
	•	Filtering and highlighting tools

⸻

Deployment

The dashboard is deployed using Streamlit Cloud.

Environment variables (secrets) are required for database connections and must be configured securely in deployment settings.
