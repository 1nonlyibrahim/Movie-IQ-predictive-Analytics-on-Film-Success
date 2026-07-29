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

            status.info("🔍 Step 1 / 4 : Checking Required Columns...")

            progress.progress(0)

            time.sleep(1.5)

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

            progress.progress(20)

            status.success("✅ Required Columns Verified")

            time.sleep(1.5)

            status.info("🧹 Step 2 / 4 : Checking Missing Values...")

            progress.progress(20)

            time.sleep(1.5)

            missing_count = int(df_local.isnull().sum().sum())

            if missing_count == 0:

                status.success("✅ No Missing Values Found")

                time.sleep(1.5)

            else:

                status.warning(
                    f"⚠ {missing_count} missing values detected."
                )

                time.sleep(1.5)

                original_rows = len(df_local)

                df_local = df_local.dropna().reset_index(drop=True)

                removed_rows = original_rows - len(df_local)

                st.session_state.df = df_local

                status.success(
                    f"✅ Removed {removed_rows} row(s) containing missing values."
                )

                time.sleep(1)

            progress.progress(40)

            # -------------------- STEP 3 : DUPLICATE VALUE CHECK --------------------

            status.info("📑 Step 3 / 4 : Checking Duplicate Values...")

            time.sleep(1.5)

            duplicate_count = int(df_local.duplicated().sum())

            if duplicate_count == 0:

                status.success("✅ No Duplicate Rows Found")

                time.sleep(1.5)

            else:

                status.warning(
                    f"⚠ {duplicate_count} duplicate row(s) detected."
                )

                time.sleep(1.5)

                original_rows = len(df_local)

                df_local = df_local.drop_duplicates().reset_index(drop=True)

                removed_rows = original_rows - len(df_local)

                st.session_state.df = df_local

                status.success(
                    f"✅ Removed {removed_rows} duplicate row(s)."
                )

                time.sleep(1.5)

            progress.progress(60)

            status.info("🔄 Step 4 / 4 : Correcting Column Data Types...")

            time.sleep(1.5)

            # Numeric Columns
            numeric_columns = [
                "budget",
                "revenue",
                "runtime",
                "popularity",
                "vote_average"
            ]

            for col in numeric_columns:

                if col in df_local.columns:

                    df_local[col] = pd.to_numeric(
                        df_local[col],
                        errors="coerce"
                    )

            # String Columns
            string_columns = [
                "title",
                "genres"
            ]

            for col in string_columns:

                if col in df_local.columns:

                    df_local[col] = (
                        df_local[col]
                        .astype(str)
                        .str.strip()
                    )

            # Remove rows that became NaN after conversion
            df_local = df_local.dropna().reset_index(drop=True)

            # Create Success Column
            if "success" not in df_local.columns:

                df_local["success"] = (
                    df_local["revenue"] >
                    df_local["budget"]
                ).astype(int)

            st.session_state.df = df_local

            progress.progress(100)

            status.success("✅ Data Types Corrected Successfully")

            time.sleep(1)

            # -------------------- VALIDATION COMPLETE --------------------

            st.session_state.df = df_local

            progress.empty()
            status.empty()

            st.success("🎉 Dataset Validation Completed Successfully!")

            st.session_state.validation_complete = True

            time.sleep(2)

            st.rerun()
        
        # Centered button to open the validation dialog
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if not st.session_state.validation_complete:
                if st.button("Perform Data Validation", type="primary", use_container_width=True):
                    st.session_state.validation_started = True
                    validation_window()

#==========================================================================================================================================================================================
#sidebar to show dataset information and statistics
#==========================================================================================================================================================================================
# ====================== SIDEBAR ======================

if st.session_state.validation_complete:

    df = st.session_state.df

    st.sidebar.title("🎬 MovieIQ")

    st.sidebar.success("Dataset Ready for Analysis")

    st.sidebar.divider()

    # ---------------- Dataset Status ---------------- #

    st.sidebar.subheader("📂 Dataset Status")

    st.sidebar.metric("Rows", f"{df.shape[0]:,}")
    st.sidebar.metric("Columns", df.shape[1])

    memory = df.memory_usage(deep=True).sum() / (1024**2)

    st.sidebar.metric(
        "Memory Usage",
        f"{memory:.2f} MB"
    )

    st.sidebar.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.sidebar.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    numeric_df = df.select_dtypes(include="number")

    outliers = 0

    for col in numeric_df.columns:

        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)

        iqr = q3 - q1

        outliers += (
            (
                (numeric_df[col] < (q1 - 1.5 * iqr)) |
                (numeric_df[col] > (q3 + 1.5 * iqr))
            )
        ).sum()

    st.sidebar.metric(
        "Outlier Count",
        int(outliers),
        help="Outliers are unusually high or low values detected using the IQR (Interquartile Range) method. They are reported for analysis but are not removed automatically."
)

    st.sidebar.metric(
        "Target Variable",
        "Available" if "success" in df.columns else "Not Created",
        help="The target variable is the output that the machine learning model predicts. In this application, a movie is labeled as Successful (1) if its revenue is greater than its budget, otherwise it is labeled as Not Successful (0)."
    )

    st.sidebar.divider()

    # ---------------- Dataset Preview ---------------- #

    with st.sidebar.expander("👀 Dataset Preview"):

        st.dataframe(
            df.head(),
            use_container_width=True
        )

    # ---------------- Column Information ---------------- #

    with st.sidebar.expander("🧾 Column Information"):

        info = pd.DataFrame({
            "Column": df.columns,
            "Datatype": df.dtypes.astype(str),
            "Missing": df.isnull().sum().values,
            "Unique": df.nunique().values
        })

        st.dataframe(
            info,
            use_container_width=True,
            height=300
        )

    # ---------------- Summary Statistics ---------------- #

    with st.sidebar.expander("📈 Summary Statistics"):

        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True,
            height=350
        )

    # ---------------- Data Cleaning Report ---------------- #

    with st.sidebar.expander("🧹 Data Cleaning Report"):

        st.success("Validation Completed")

        st.write(f"**Final Shape :** {df.shape}")

        st.write(f"**Missing Values :** {int(df.isnull().sum().sum())}")

        st.write(f"**Duplicate Rows :** {int(df.duplicated().sum())}")

        st.write(
            f"**Success Column :** {'Yes' if 'success' in df.columns else 'No'}"
        )

if st.session_state.validation_complete:

    df = st.session_state.df
    #==========================================================================================================================================================================================
    #KPI cards
    #==========================================================================================================================================================================================
    def format_currency_indian(num):
        num = float(num)

        if abs(num) >= 1e7:      # 1 Crore
            return f"₹{num/1e7:,.2f} Cr"

        elif abs(num) >= 1e5:    # 1 Lakh
            return f"₹{num/1e5:,.2f} L"

        elif abs(num) >= 1e3:    # 1 Thousand
            return f"₹{num/1e3:,.2f} K"

        return f"₹{num:,.2f}"

    st.markdown("## 📊 Executive Overview")

    # ---------- Calculate KPIs ----------

    total_movies = len(df)

    total_revenue = df["revenue"].sum()

    total_budget = df["budget"].sum()

    avg_rating = df["vote_average"].mean()

    avg_popularity = df["popularity"].mean()

    avg_runtime = df["runtime"].mean()

    if "success" in df.columns:
        success_rate = df["success"].mean() * 100
    else:
        success_rate = ((df["revenue"] > df["budget"]).mean()) * 100

    # Handle multiple genres separated by |
    genre_series = (
        df["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )

    most_common_genre = (
        genre_series.mode().iloc[0]
        if not genre_series.mode().empty
        else "N/A"
    )

    # ---------- KPI Row 1 ----------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🎬 Total Movies",
            f"{total_movies:,}"
        )

    with c2:
        st.metric(
            "💰 Total Revenue",
            format_indian(total_revenue)
        )

    with c3:
        st.metric(
            "💸 Total Budget",
            format_indian(total_budget)
        )

    with c4:
        st.metric(
            "⭐ Average Rating",
            f"{avg_rating:.2f}/10"
        )

    # ---------- KPI Row 2 ----------

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric(
            "📈 Avg Popularity",
            f"{avg_popularity:.2f}"
        )

    with c6:
        st.metric(
            "⏱ Avg Runtime",
            f"{avg_runtime:.1f} min"
        )

    with c7:
        st.metric(
            "✅ Success Rate",
            f"{success_rate:.1f}%"
        )

    with c8:
        st.markdown("""<style>[data-testid="stMetricValue"]{ font-size:22px !important;}</style>""", unsafe_allow_html=True)
        st.metric(
            "🎭 Most Common Genre",
            most_common_genre
        )

    st.divider()