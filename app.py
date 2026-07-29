import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import joblib
import time

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "MovieIQ - Intelligent Movie Dataset Analysis & Prediction Platform"
    }
)
REQUIRED_COLUMNS = [
    "title",
    "budget",
    "revenue",
    "genres",
    "runtime",
    "popularity",
    "vote_average"
]

NUMERIC_COLUMNS = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_average"
]
if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False

if "cleaning_done" not in st.session_state:
    st.session_state.cleaning_done = False

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "df" not in st.session_state:
    st.session_state.df = None


def bytes_to_mb(size):
    return round(size / (1024 * 1024), 2)


def memory_usage(df):
    return bytes_to_mb(df.memory_usage(deep=True).sum())
def check_required_columns(df):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return missing
def count_missing(df):
    return int(df.isnull().sum().sum())
def count_duplicates(df):
    return int(df.duplicated().sum())
def count_outliers(df):

    total = 0

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            total += (
                (df[column] < lower) |
                (df[column] > upper)
            ).sum()

    return int(total)
def invalid_values(df):

    invalid = {}

    if "budget" in df.columns:
        invalid["budget"] = (df["budget"] <= 0).sum()

    if "revenue" in df.columns:
        invalid["revenue"] = (df["revenue"] <= 0).sum()

    if "runtime" in df.columns:
        invalid["runtime"] = (df["runtime"] <= 0).sum()

    if "popularity" in df.columns:
        invalid["popularity"] = (df["popularity"] < 0).sum()

    if "vote_average" in df.columns:
        invalid["vote_average"] = (
            (df["vote_average"] < 0) |
            (df["vote_average"] > 10)
        ).sum()

    return invalid
def create_target(df):

    if "success" not in df.columns:

        df["success"] = np.where(
            df["revenue"] > df["budget"],
            1,
            0
        )

    return df
def main():

    st.markdown(
    "<h1 style='text-align: center;'>MOVIEIQ - INTELLIGENT MOVIE DATASET ANALYSIS & PREDICTION</h1>", 
    unsafe_allow_html=True
)

    st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.85em;'>Intelligent Movie Dataset Analysis & Prediction Platform</p>", 
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()

#==========================================================================================================================================================================================
#upload the dataset
#==========================================================================================================================================================================================

st.markdown("""
<style>
.block-container{padding-top:2rem;}
h1{text-align:center;}
[data-testid="stFileUploader"]{width:70%;margin:auto;}
[data-testid="stFileUploader"] section{
border:2px dashed #4F8BF9;
border-radius:18px;
padding:55px;
background:#181818;
transition:.3s;
}
[data-testid="stFileUploader"] section:hover{
border-color:#78A9FF;
background:#222;
transform:scale(1.01);
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & Drop your CSV here or Click to Upload",
    type="csv",
    help="Only CSV files are supported."
)

if uploaded_file:
    progress = st.progress(0)
    status = st.empty()

    steps = [
        "📂 Reading Dataset...",
        "🔍 Validating Structure...",
        "🧹 Detecting Missing Values...",
        "📑 Checking Duplicate Records...",
        "⚙️ Preparing Dashboard...",
        "✅ Finalizing..."
    ]

    for i, step in enumerate(steps):
        status.info(step)
        progress.progress(int((i + 1) / len(steps) * 100))
        time.sleep(0.5)

    df = pd.read_csv(uploaded_file)

    st.toast("🎉 Dataset Uploaded Successfully!", icon="✅")
    st.success("Dataset is ready for analysis.")