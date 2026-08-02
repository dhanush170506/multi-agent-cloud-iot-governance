import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="TAPNA Dashboard", layout="wide")

experiment_results = pd.read_csv("data/experiment_results.csv")
tapna_log = pd.read_csv("data/tapna_log.csv")

strategies = sorted(experiment_results["strategy"].unique())
selected_strategies = st.sidebar.multiselect(
    "Strategies",
    strategies,
    default=strategies,
)

filtered_results = experiment_results[
    experiment_results["strategy"].isin(selected_strategies)
]

st.title("TAPNA Experiment Dashboard")

cigi_chart = px.line(
    filtered_results,
    x="tick",
    y="CIGI",
    color="strategy",
)
st.plotly_chart(cigi_chart, use_container_width=True)

tapna_weights_chart = px.line(
    tapna_log,
    x="tick",
    y="wS",
)
st.plotly_chart(tapna_weights_chart, use_container_width=True)

summary = (
    experiment_results
    .groupby("strategy", as_index=False)
    .agg(
        Average_CIGI=("CIGI", "mean"),
        Average_response_lag=("response_lag", "mean"),
    )
)

st.dataframe(summary, use_container_width=True, hide_index=True)
