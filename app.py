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

    st.title("🎬 MovieIQ")

    st.caption(
        "Intelligent Movie Dataset Analysis & Prediction Platform"
    )


if __name__ == "__main__":
    main()