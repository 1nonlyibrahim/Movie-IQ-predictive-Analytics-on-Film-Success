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
    return int(df.duplicated(keep=False).sum())
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
    width:{"70%" if not st.session_state.uploaded else "45%"};
    margin:auto;
    transition:all .4s ease;
}}

[data-testid="stFileUploader"] section{{
    border:2px dashed #4F8BF9;
    border-radius:18px;
    padding:{"60px" if not st.session_state.uploaded else "18px"};
    background:#181818;
    transition:.3s;
}}

[data-testid="stFileUploader"] section:hover{{
    border-color:#78A9FF;
    background:#222;
}}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & Drop your CSV here or Click to Upload",
    type=["csv"],
    help="Only CSV files are supported."
)

if uploaded_file and not st.session_state.uploaded:

    loader = st.empty()
    progress = st.progress(0)

    steps = [
        "📂 Reading Dataset...",
        "📄 Parsing CSV...",
        "🔍 Validating File...",
        "⚙ Preparing Workspace...",
        "🚀 Almost Ready..."
    ]

    # Show a staged progress animation while attempting to read once
    try:
        # initial stage
        for i, s in enumerate(steps):
            loader.info(s)
            progress.progress(int((i + 1) / len(steps) * 100))
            time.sleep(0.4)

        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.uploaded = True

        loader.empty()
        progress.empty()

        success = st.empty()
        success.success("✅ Dataset Uploaded Successfully! Preparing for Validation...")
        time.sleep(1.5)
        success.empty()

    except Exception as e:
        loader.error(f"Unable to read CSV.\n{e}")
        progress.empty()
        # don't set uploaded flag or session df on failure

#==========================================================================================================================================================================================
#check for any missing columns in the dataset
#==========================================================================================================================================================================================

REQUIRED_COLUMNS = [
    "title",
    "budget",
    "revenue",  
    "genres",
    "runtime",
    "popularity",
    "vote_average"
]

if "validation_started" not in st.session_state:
    st.session_state.validation_started = False

if "validation_complete" not in st.session_state:
    st.session_state.validation_complete = False

if st.session_state.uploaded:

    df = st.session_state.df

    st.divider()

    if not st.session_state.validation_started:

        @st.dialog("🔍 Dataset Validation", width="large")
        def validation_window():
            progress = st.progress(0)
            status = st.empty()

            status.info("🔍 Step 1 / 3 : Checking Required Columns...")

            progress.progress(15)

            time.sleep(0.8)

            # Safely fetch the dataframe from session state to avoid
            # any local variable shadowing / UnboundLocalError.
            df_local = st.session_state.get("df")

            if df_local is None:
                st.error("No dataset found in session. Please upload a CSV first.")
                st.stop()

            missing_columns = [
                col
                for col in REQUIRED_COLUMNS
                if col not in df_local.columns
            ]

            progress.progress(33)

            if missing_columns:

                status.error("❌ Required Columns Missing")

                st.error(
                    "The uploaded dataset cannot be analyzed because required columns are missing."
                )

                st.dataframe(
                    pd.DataFrame(
                        {"Missing Columns": missing_columns}
                    ),
                    use_container_width=True
                )

                st.stop()

            progress.progress(100)

            status.success("✅ Required Columns Verified")

            time.sleep(1)

            status.info("🧹 Step 2 / 3 : Checking Missing Values...")

            progress.progress(45)

            time.sleep(0.8)

            missing_count = int(df_local.isnull().sum().sum())

            if missing_count == 0:

                progress.progress(66)

                status.success("✅ No Missing Values Found")

                time.sleep(1)

            else:

                status.warning(
                    f"⚠ {missing_count} missing values detected."
                )

                time.sleep(0.8)

                original_rows = len(df_local)

                df_local = df_local.dropna().reset_index(drop=True)

                removed_rows = original_rows - len(df_local)

                st.session_state.df = df_local

                progress.progress(66)

                status.success(
                    f"✅ Removed {removed_rows} row(s) containing missing values."
                )

                time.sleep(1)

            progress.progress(100)
            status.success("✅ Missing Value Validation Completed")

            time.sleep(1)

            progress.empty()
            status.empty()

            st.success("Step 2 Completed")

            st.session_state.validation_complete = True

            st.info(
                "Next Step → Missing Value Detection"
            )
        
        # Centered button to open the validation dialog
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Perform Data Validation", type="primary", use_container_width=True):
                st.session_state.validation_started = True
                validation_window()
#==========================================================================================================================================================================================
#check for any duplicate values in the dataset
#==========================================================================================================================================================================================

#==========================================================================================================================================================================================
#validation completion and analysis readiness
#==========================================================================================================================================================================================