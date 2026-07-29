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


if __name__ == "__main__":
    main()

#==========================================================================================================================================================================================
#upload the dataset
#==========================================================================================================================================================================================
if "df" not in st.session_state:
    st.session_state.df = None

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

st.markdown(f"""
<style>
.block-container{{padding-top:2rem;}}
[data-testid="stFileUploader"]{{
    width:{"70%" if not st.session_state.uploaded else "35%"};
    margin:auto;
    transition:all .4s ease;
}}

[data-testid="stFileUploader"] section{{
    border:2px dashed #4F8BF9;
    border-radius:18px;
    padding:{"60px" if not st.session_state.uploaded else "15px"};
    background:#181818;
    transition:.3s;
}}

[data-testid="stFileUploader"] section:hover{{
    border-color:#78A9FF;
    background:#222;
}}
</style>
""", unsafe_allow_html=True)

if not st.session_state.uploaded:

    uploaded_file = st.file_uploader(
        "Drag & Drop your CSV here or Click to Upload",
        type=["csv"],
        help="Only CSV files are supported."
    )

if uploaded_file and not st.session_state.uploaded:

        loader=st.empty()
        progress=st.progress(0)

        steps=[
            "📂 Reading Dataset...",
            "📄 Parsing CSV...",
            "🔍 Validating File...",
            "⚙ Preparing Workspace...",
            "🚀 Almost Ready..."
        ]

        for i,s in enumerate(steps):
            loader.info(s)
            progress.progress(int((i+1)/len(steps)*100))
            time.sleep(0.6)

        try:
            df=pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.uploaded=True

            loader.empty()
            progress.empty()

            st.toast("🎉 Dataset Uploaded Successfully!",icon="✅")
            st.success("Dataset uploaded successfully. Preparing analysis dashboard...")


        except Exception as e:
            loader.error(f"Unable to read CSV.\n{e}")

        msg = st.empty()

        msg.success("✅ Dataset Uploaded Successfully!")

        time.sleep(3)

        msg.empty()

        st.rerun()

#==========================================================================================================================================================================================
#upload the dataset
#==========================================================================================================================================================================================

