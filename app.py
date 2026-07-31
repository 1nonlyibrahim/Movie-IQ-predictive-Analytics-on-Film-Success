import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
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

            st.session_state.show_validation_dialog = False

            time.sleep(1.5)

            st.rerun()
        
        # Centered button to open the validation dialog
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if (
                not st.session_state.validation_complete
                and
                not st.session_state.show_validation_dialog
            ):

                if st.button(
                    "Perform Data Validation",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.show_validation_dialog = True
                    st.rerun()
if "validation_started" not in st.session_state:
    st.session_state.validation_started = False

if "validation_complete" not in st.session_state:
    st.session_state.validation_complete = False

if "show_validation_dialog" not in st.session_state:
    st.session_state.show_validation_dialog = False
if st.session_state.show_validation_dialog:
    validation_window()

#==================================================================================================================================================================================================================================
#🎬 Movie Success Predictor
#==================================================================================================================================================================================================================================
if st.session_state.validation_complete:
    st.markdown("""
        <style>
        .mspbutton {
        font-size: 22px;
        font-weight: bold;
        color: #4F8BF9;
        text-align: center;
        padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )   

    @st.dialog("🎬 Movie Success Prediction", width="large")
    def prediction_window():
        
        df = st.session_state.df

        budget = st.number_input("Budget", min_value=0.0)
        runtime = st.number_input("Runtime", min_value=1)
        popularity = st.number_input("Popularity", min_value=0.0, max_value=100.0)
        vote_average = st.slider("Vote Average", 0.0, 10.0, 7.0)
        genre = st.selectbox(
            "Genre",
            sorted(df["genres"].dropna().unique())
        )

        if st.button("🚀 Predict Success"):
            if "model" not in st.session_state:
                st.error(
                    "Please open the Machine Learning section first."
                )
                st.stop()

            model = st.session_state.model

            input_data = pd.DataFrame(
                [[budget, runtime, popularity]],
                columns=[
                    "budget",
                    "runtime",
                    "popularity"
                ]
            )

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            confidence = max(probability, 1 - probability)

            if prediction == 1:
                st.success("🎉 Movie is predicted to be Successful!")
            else:
                st.error("❌ Movie is predicted to be Unsuccessful.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Success Probability",
                    f"{probability*100:.2f}%"
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("""<style>
div.stButton > button {
    height: 70px !important;
    font-size: 50px !important;
    font-weight: 700 !important;
    border-radius: 15px !important;
}
</style>""", unsafe_allow_html=True)
    st.markdown('<div class="mspbutton  ">', unsafe_allow_html=True)
    if st.button(
        "🎬 Movie Success Predictor",
        use_container_width=True,   
        type="primary"
    ):
        prediction_window()
st.divider()

#===============================================================================================================================================================================================================================
#Analysis filters 
#===============================================================================================================================================================================================================================
if st.session_state.validation_complete:
    with st.expander("🔍 Analysis Filters", expanded=False):
        st.subheader("🔍 Analysis Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_genres = st.multiselect(
                "🎭 Genre",
                sorted(df["genres"].dropna().unique())
            )

        with col2:
            budget_range = st.slider(
                "💰 Budget",
                float(df["budget"].min()),
                float(df["budget"].max()),
                (
                    float(df["budget"].min()),
                    float(df["budget"].max())
                )
            )

        with col3:
            revenue_range = st.slider(
                "📈 Revenue",
                float(df["revenue"].min()),
                float(df["revenue"].max()),
                (
                    float(df["revenue"].min()),
                    float(df["revenue"].max())
                )
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            runtime_range = st.slider(
                "⏱ Runtime",
                int(df["runtime"].min()),
                int(df["runtime"].max()),
                (
                    int(df["runtime"].min()),
                    int(df["runtime"].max())
                )
            )

        with col5:
            rating_range = st.slider(
                "⭐ Vote Average",
                float(df["vote_average"].min()),
                float(df["vote_average"].max()),
                (
                    float(df["vote_average"].min()),
                    float(df["vote_average"].max())
                )
            )

        with col6:
            success_filter = st.selectbox(
                "✅ Success",
                [
                    "All",
                    "Successful",
                    "Not Successful"
                ]
            )

        movie_search = st.text_input(
            "🎬 Search Movie"
        )
st.divider()
#==========================================================================================================================================================================================
#sidebar to show dataset information and statistics
#==========================================================================================================================================================================================
# ====================== SIDEBAR ======================

if st.session_state.validation_complete:

    df = st.session_state.df

    st.sidebar.title("📊 Dashboard Navigation")

    analysis_section = st.sidebar.radio(
        "Select Analysis",
        [
            "📊 Executive Overview",
            "📈 Distribution Analysis",
            "🎭 Genre Analysis",
            "💰 Financial Analysis",
            "⭐ Rating & Popularity Analysis",
            "🔗 Correlation Analysis",
            "📊 Statistical Analysis",
            "💡 Business Insights",
            "📝 Business Recommendations",
            "🤖 Machine Learning"
        ]
    )
    st.sidebar.divider()

    st.sidebar.title("📂 Dataset Info")

    

    # ---------------- Dataset Status ---------------- #

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

    st.sidebar.divider()

    st.sidebar.link_button(
    "💻 View Source Code on GitHub",
    "https://github.com/1nonlyibrahim/Movie-IQ-predictive-Analytics-on-Film-Success/blob/d90ceae95fa143d406ccf416b7b2c8d8e0d0c899/app.py",
    use_container_width=True
    )

if st.session_state.validation_complete:

    if analysis_section == "📊 Executive Overview":
        df = st.session_state.df
        #==========================================================================================================================================================================================
        #KPI cards
        #==========================================================================================================================================================================================
        def format_currency_indian(num):
            num = float(num)
            sign = "-" if num < 0 else ""
            num = abs(num)

            if num >= 1e7:  # Crore
                return f"{sign}₹{num/1e7:,.2f} Cr"

            elif num >= 1e5:  # Lakh
                return f"{sign}₹{num/1e5:,.2f} L"

            else:
                # Indian comma format
                integer, decimal = f"{num:.2f}".split(".")

                if len(integer) > 3:
                    last3 = integer[-3:]
                    rest = integer[:-3]

                    parts = []
                    while len(rest) > 2:
                        parts.insert(0, rest[-2:])
                        rest = rest[:-2]

                    if rest:
                        parts.insert(0, rest)

                    integer = ",".join(parts + [last3])

                return f"{sign}₹{integer}.{decimal}"

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
                format_currency_indian(total_revenue)
            )

        with c3:
            st.metric(
                "💸 Total Budget",
                format_currency_indian(total_budget)
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

    if analysis_section == "📈 Distribution Analysis":

        #==========================================================================================================================================================================================
        #               📈  DISTRIBUTION ANALYSIS
        #==========================================================================================================================================================================================

        st.markdown("## 📊 Distribution Analysis")

        variables = [
            ("budget", "💸 Budget Distribution"),
            ("revenue", "💰 Revenue Distribution"),
            ("runtime", "⏱ Runtime Distribution"),
            ("popularity", "📈 Popularity Distribution"),
            ("vote_average", "⭐ Vote Average Distribution")
        ]

        for column, title in variables:

            st.markdown(f"### {title}")

            col1, col2 = st.columns(2)

            # ---------------- Histogram ---------------- #

            with col1:

                plot_df = df.copy()

                if column in ["budget", "revenue"]:
                    plot_df[column] = plot_df[column] / 1e7
                    x_title = f"{column.replace('_', ' ').title()} (₹ Crore)"
                else:
                    x_title = column.replace("_", " ").title()

                fig = px.histogram(
                    plot_df,
                    x=column,
                    nbins=30,
                    title=f"{column.replace('_',' ').title()} Distribution",
                    color_discrete_sequence=["#4F8BF9"]
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title=x_title,
                    yaxis_title="No. of Movies"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ---------------- Box Plot ---------------- #

            with col2:

                fig = px.box(
                    plot_df,
                    y=column,
                    title=f"{column.replace('_',' ').title()} Box Plot",
                    color_discrete_sequence=["#FF6B6B"]
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis_title=x_title
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ---------------- Quick Summary ---------------- #

            if column in ["budget", "revenue"]:
                avg = format_currency_indian(df[column].mean())
                median = format_currency_indian(df[column].median())
                minimum = format_currency_indian(df[column].min())
                maximum = format_currency_indian(df[column].max())
                std = format_currency_indian(df[column].std())

            elif column == "runtime":
                avg = f"{df[column].mean():.1f} min"
                median = f"{df[column].median():.1f} min"
                minimum = f"{df[column].min():.1f} min"
                maximum = f"{df[column].max():.1f} min"
                std = f"{df[column].std():.1f} min"

            elif column == "vote_average":
                avg = f"{df[column].mean():.2f}/10"
                median = f"{df[column].median():.2f}/10"
                minimum = f"{df[column].min():.2f}/10"
                maximum = f"{df[column].max():.2f}/10"
                std = f"{df[column].std():.2f}"

            else:   # popularity and any other numeric columns
                avg = f"{df[column].mean():,.2f}"
                median = f"{df[column].median():,.2f}"
                minimum = f"{df[column].min():,.2f}"
                maximum = f"{df[column].max():,.2f}"
                std = f"{df[column].std():,.2f}"

            st.info(    
                f"""
        **📌 Distribution Summary**

        • The average **{column.replace('_', ' ').title()}** across all movies is **{avg}**.

        • Half of the movies have a **{column.replace('_', ' ').title()}** below **{median}**, while the other half are above it.

        • The recorded values range from **{minimum}** to **{maximum}**.

        • The overall spread of the data is **{std}**, indicating how much the values vary from the average.
        """
            )

            st.divider()

    if analysis_section == "🎭 Genre Analysis":

        if st.session_state.get("validation_complete", False):

            df = st.session_state.df
            # ==========================================================
            #                   🎭 GENRE ANALYSIS
            # ==========================================================

            st.markdown("## 🎭 Genre Analysis")

            genre_df = df.copy()

            genre_df["genres"] = genre_df["genres"].fillna("Unknown")

            genre_df = (
                genre_df
                .assign(genres=genre_df["genres"].str.split("|"))
                .explode("genres")
            )

            genre_df["genres"] = genre_df["genres"].str.strip()

            genre_count = (
                genre_df["genres"]
                .value_counts()
                .reset_index()
            )

        genre_count.columns = ["Genre", "Movies"]

        fig = px.bar(
            genre_count,
            x="Genre",
            y="Movies",
            color="Movies",
            color_continuous_scale="Blues",
            text_auto=True,
            title="Movie Count by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Genre",
            yaxis_title="Number of Movies",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        genre_revenue = (
            genre_df
            .groupby("genres")["revenue"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        genre_revenue["Revenue (Cr)"] = genre_revenue["revenue"] / 1e7

        fig = px.bar(
            genre_revenue,
            x="genres",
            y="Revenue (Cr)",
            color="Revenue (Cr)",
            color_continuous_scale="Greens",
            text_auto=".2f",
            title="Average Revenue by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Genre",
            yaxis_title="Average Revenue (₹ Crore)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        genre_rating = (
            genre_df
            .groupby("genres")["vote_average"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            genre_rating,
            x="genres",
            y="vote_average",
            color="vote_average",
            color_continuous_scale="Purples",
            text_auto=".2f",
            title="Average Rating by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Genre",
            yaxis_title="Average Rating",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        genre_budget = (
            genre_df
            .groupby("genres")["budget"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        genre_budget["Budget (Cr)"] = genre_budget["budget"] / 1e7

        fig = px.bar(
            genre_budget,
            x="genres",
            y="Budget (Cr)",
            color="Budget (Cr)",
            color_continuous_scale="Oranges",
            text_auto=".2f",
            title="Average Budget by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Genre",
            yaxis_title="Average Budget (₹ Crore)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        SUCCESS_THRESHOLD = 7.0

        genre_df["Successful"] = genre_df["vote_average"] >= SUCCESS_THRESHOLD

        success_rate = (
            genre_df
            .groupby("genres")["Successful"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        success_rate.columns = ["Genre", "Success Rate (%)"]

        fig = px.bar(
            success_rate,
            x="Genre",
            y="Success Rate (%)",
            color="Success Rate (%)",
            color_continuous_scale="Viridis",
            text_auto=".1f",
            title="Success Rate by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Genre",
            yaxis_title="Success Rate (%)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    #                 💰 FINANCIAL ANALYSIS
    # ==========================================================

    if analysis_section == "💰 Financial Analysis":

        st.caption("Analyze the financial performance of movies based on revenue, budget, profit and return on investment.")

        financial_df = df.copy()

        financial_df["Profit"] = financial_df["revenue"] - financial_df["budget"]

        financial_df["ROI (%)"] = np.where(
            financial_df["budget"] > 0,
            ((financial_df["revenue"] - financial_df["budget"]) / financial_df["budget"]) * 100,
            0
        )

        financial_df["Budget (Cr)"] = financial_df["budget"] / 1e7
        financial_df["Revenue (Cr)"] = financial_df["revenue"] / 1e7
        financial_df["Profit (Cr)"] = financial_df["Profit"] / 1e7

        # =====================================================
        # Top 10 Highest Revenue Movies
        # =====================================================

        st.subheader("🎬 Top 10 Highest Revenue Movies")

        top_revenue = (
            financial_df
            .sort_values("revenue", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_revenue,
            x="title",
            y="Revenue (Cr)",
            color="Revenue (Cr)",
            color_continuous_scale="Greens",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Movie",
            yaxis_title="Revenue (₹ Crore)",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Top 10 Highest Budget Movies
        # =====================================================

        st.subheader("💸 Top 10 Highest Budget Movies")

        top_budget = (
            financial_df
            .sort_values("budget", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_budget,
            x="title",
            y="Budget (Cr)",
            color="Budget (Cr)",
            color_continuous_scale="Oranges",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Movie",
            yaxis_title="Budget (₹ Crore)",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Budget vs Revenue
        # =====================================================

        st.subheader("📈 Budget vs Revenue")

        fig = px.scatter(
            financial_df,
            x="Budget (Cr)",
            y="Revenue (Cr)",
            hover_name="title",
            color="vote_average",
            size="popularity",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Budget (₹ Crore)",
            yaxis_title="Revenue (₹ Crore)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Top 10 Most Profitable Movies
        # =====================================================

        st.subheader("💵 Top 10 Most Profitable Movies")

        top_profit = (
            financial_df
            .sort_values("Profit", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_profit,
            x="title",
            y="Profit (Cr)",
            color="Profit (Cr)",
            color_continuous_scale="Tealgrn",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Movie",
            yaxis_title="Profit (₹ Crore)",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Top 10 ROI Movies
        # =====================================================

        st.subheader("📊 Top 10 ROI (%) Movies")

        roi_df = (
            financial_df[financial_df["budget"] > 0]
            .sort_values("ROI (%)", ascending=False)
            .head(10)
        )

        fig = px.bar(
            roi_df,
            x="title",
            y="ROI (%)",
            color="ROI (%)",
            color_continuous_scale="Plasma",
            text_auto=".1f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Movie",
            yaxis_title="ROI (%)",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, use_container_width=True)


    # ==========================================================
    #          ⭐ RATING & POPULARITY ANALYSIS
    # ==========================================================

    if analysis_section == "⭐ Rating & Popularity Analysis":

        st.caption(
            "Explore audience reception, popularity trends and their relationship with revenue."
        )

        analysis_df = df.copy()

        # =====================================================
        # Rating Distribution
        # =====================================================

        st.subheader("⭐ Rating Distribution")

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                analysis_df,
                x="vote_average",
                nbins=20,
                color_discrete_sequence=["#4F8BF9"]
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Rating",
                yaxis_title="Number of Movies"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            fig = px.box(
                analysis_df,
                y="vote_average",
                color_discrete_sequence=["#FF6B6B"]
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                yaxis_title="Rating"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Popularity Distribution
        # =====================================================

        st.subheader("🔥 Popularity Distribution")

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                analysis_df,
                x="popularity",
                nbins=30,
                color_discrete_sequence=["#00CC96"]
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Popularity",
                yaxis_title="Number of Movies"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            fig = px.box(
                analysis_df,
                y="popularity",
                color_discrete_sequence=["#FFA15A"]
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                yaxis_title="Popularity"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Popularity vs Rating
        # =====================================================

        st.subheader("📈 Popularity vs Rating")

        fig = px.scatter(
            analysis_df,
            x="popularity",
            y="vote_average",
            hover_name="title",
            size="revenue",
            color="vote_average",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Popularity",
            yaxis_title="Rating"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Revenue vs Rating
        # =====================================================

        st.subheader("💰 Revenue vs Rating")

        revenue_df = analysis_df.copy()
        revenue_df["Revenue (Cr)"] = revenue_df["revenue"] / 1e7

        fig = px.scatter(
            revenue_df,
            x="vote_average",
            y="Revenue (Cr)",
            hover_name="title",
            size="popularity",
            color="Revenue (Cr)",
            color_continuous_scale="Greens"
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Rating",
            yaxis_title="Revenue (₹ Crore)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Popularity vs Revenue
        # =====================================================

        st.subheader("🔥 Popularity vs Revenue")

        fig = px.scatter(
            revenue_df,
            x="popularity",
            y="Revenue (Cr)",
            hover_name="title",
            size="vote_average",
            color="Revenue (Cr)",
            color_continuous_scale="Plasma"
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Popularity",
            yaxis_title="Revenue (₹ Crore)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    #                🔗 CORRELATION ANALYSIS
    # ==========================================================

    if analysis_section == "🔗 Correlation Analysis":

        st.caption(
            "Analyze relationships between numerical variables to identify strong positive and negative correlations."
        )

        corr_df = df.copy()

        numeric_df = corr_df.select_dtypes(include="number")

        correlation_matrix = numeric_df.corr(numeric_only=True)

        # =====================================================
        # Correlation Heatmap
        # =====================================================

        st.subheader("🔥 Correlation Heatmap")

        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        fig.update_layout(
            template="plotly_dark",
            height=650
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =====================================================
        # Correlation Matrix
        # =====================================================

        st.subheader("📋 Correlation Matrix")

        st.dataframe(
            correlation_matrix.style.format("{:.2f}"),
            use_container_width=True
        )

        st.divider()

        # =====================================================
        # Strongest Positive & Negative Correlation
        # =====================================================

        corr_pairs = (
            correlation_matrix
            .where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )

        corr_pairs.columns = [
            "Variable 1",
            "Variable 2",
            "Correlation"
        ]

        strongest_positive = corr_pairs.loc[
            corr_pairs["Correlation"].idxmax()
        ]

        strongest_negative = corr_pairs.loc[
            corr_pairs["Correlation"].idxmin()
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.success("📈 Strongest Positive Correlation")

            st.metric(
                label=f"{strongest_positive['Variable 1']} ↔ {strongest_positive['Variable 2']}",
                value=f"{strongest_positive['Correlation']:.2f}"
            )

        with col2:

            st.error("📉 Strongest Negative Correlation")

            st.metric(
                label=f"{strongest_negative['Variable 1']} ↔ {strongest_negative['Variable 2']}",
                value=f"{strongest_negative['Correlation']:.2f}"
            )

    # ==========================================================
    #               📊 STATISTICAL ANALYSIS
    # ==========================================================

    if analysis_section == "📊 Statistical Analysis":

        st.caption(
            "Perform statistical hypothesis tests to identify significant relationships in the dataset."
        )

        stats_df = df.copy()

        # ------------------------------------------------------
        # Define Success
        # ------------------------------------------------------

        SUCCESS_THRESHOLD = 7.0

        stats_df["Successful"] = (
            stats_df["vote_average"] >= SUCCESS_THRESHOLD
        )

        # ------------------------------------------------------
        # T-Test
        # ------------------------------------------------------

        st.subheader("📈 Independent T-Test")

        success_revenue = stats_df.loc[
            stats_df["Successful"],
            "revenue"
        ].dropna()

        failure_revenue = stats_df.loc[
            ~stats_df["Successful"],
            "revenue"
        ].dropna()

        t_stat, t_pvalue = ttest_ind(
            success_revenue,
            failure_revenue,
            equal_var=False
        )

        decision = (
            "Reject Null Hypothesis"
            if t_pvalue < 0.05
            else "Fail to Reject Null Hypothesis"
        )

        interpretation = (
            "There is a statistically significant difference in average revenue between successful and unsuccessful movies."
            if t_pvalue < 0.05
            else
            "No statistically significant difference in average revenue was found."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("T Statistic", f"{t_stat:.3f}")

        with col2:
            st.metric("P-value", f"{t_pvalue:.5f}")

        st.info(f"**Decision:** {decision}")

        st.success(f"**Interpretation:** {interpretation}")

        st.divider()

        # ------------------------------------------------------
        # Chi-Square Test
        # ------------------------------------------------------

        st.subheader("🎭 Chi-Square Test")

        genre_df = stats_df.copy()

        genre_df["genres"] = genre_df["genres"].fillna("Unknown")

        genre_df = (
            genre_df
            .assign(genres=genre_df["genres"].str.split("|"))
            .explode("genres")
        )

        contingency_table = pd.crosstab(
            genre_df["genres"],
            genre_df["Successful"]
        )

        chi2, chi_pvalue, dof, expected = chi2_contingency(
            contingency_table
        )

        decision = (
            "Reject Null Hypothesis"
            if chi_pvalue < 0.05
            else "Fail to Reject Null Hypothesis"
        )

        interpretation = (
            "Movie genre and success are statistically associated."
            if chi_pvalue < 0.05
            else
            "Movie genre and success appear to be independent."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Chi-Square Statistic", f"{chi2:.3f}")

        with col2:
            st.metric("P-value", f"{chi_pvalue:.5f}")

        st.info(f"**Decision:** {decision}")

        st.success(f"**Interpretation:** {interpretation}")

    # ==========================================================
    #                  🤖 MACHINE LEARNING
    # ==========================================================

    if analysis_section == "🤖 Machine Learning":

        st.caption(
            "Train a Random Forest model to classify whether a movie is successful."
        )

        ml_df = df.copy()

        # ------------------------------------------------------
        # Target Variable
        # ------------------------------------------------------

        SUCCESS_THRESHOLD = 7.0

        ml_df["Success"] = (
            ml_df["vote_average"] >= SUCCESS_THRESHOLD
        ).astype(int)

        # ------------------------------------------------------
        # Features
        # ------------------------------------------------------

        FEATURES = [
            "budget",
            "runtime",
            "popularity"
        ]

        X = ml_df[FEATURES]

        y = ml_df["Success"]

        # ------------------------------------------------------
        # Train Test Split
        # ------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # ------------------------------------------------------
        # Model
        # ------------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        st.session_state.model = model

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions)

        recall = recall_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        roc = roc_auc_score(y_test, probabilities)

        st.subheader("📈 Model Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{accuracy:.2%}")

        c2.metric("Precision", f"{precision:.2%}")

        c3.metric("Recall", f"{recall:.2%}")

        c1, c2 = st.columns(2)

        c1.metric("F1 Score", f"{f1:.2%}")

        c2.metric("ROC AUC", f"{roc:.2%}")

        st.divider()

        # ------------------------------------------------------
        # Confusion Matrix
        # ------------------------------------------------------

        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_test, predictions)

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicted",
                y="Actual",
                color="Count"
            ),
            x=["Not Successful", "Successful"],
            y=["Not Successful", "Successful"]
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ------------------------------------------------------
        # Classification Report
        # ------------------------------------------------------

        st.subheader("📋 Classification Report")

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
            .round(3)
        )

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.divider()

        # ------------------------------------------------------
        # Feature Importance
        # ------------------------------------------------------

        st.subheader("⭐ Feature Importance")

        importance_df = pd.DataFrame({

            "Feature": FEATURES,

            "Importance": model.feature_importances_

        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        fig = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Importance",
            color_continuous_scale="Viridis",
            text_auto=".3f"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.success("✅ Model Used: Random Forest Classifier")

    #==================================================================================================================================================================================================================================
    # 💡 Business Insights
    #==================================================================================================================================================================================================================================

    if analysis_section == "💡 Business Insights":

        st.caption(
            "Automatically generated insights based on the analyzed movie dataset."
        )

        insight_df = df.copy()

        # ---------------------------------------------------------
        # Pre-processing
        # ---------------------------------------------------------

        insight_df["Profit"] = (
            insight_df["revenue"] - insight_df["budget"]
        )

        insight_df["ROI"] = np.where(
            insight_df["budget"] > 0,
            ((insight_df["revenue"] - insight_df["budget"])
            / insight_df["budget"]) * 100,
            0
        )

        SUCCESS_THRESHOLD = 7.0

        insight_df["Successful"] = np.where(
            insight_df["vote_average"] >= SUCCESS_THRESHOLD,
            "Successful",
            "Not Successful"
        )

        genre_df = insight_df.copy()

        genre_df["genres"] = genre_df["genres"].str.split("|")

        genre_df = genre_df.explode("genres")

        # ---------------------------------------------------------
        # Highest Revenue Genre
        # ---------------------------------------------------------

        highest_revenue_genre = (
            genre_df.groupby("genres")["revenue"]
            .mean()
            .idxmax()
        )

        # ---------------------------------------------------------
        # Highest Rated Genre
        # ---------------------------------------------------------

        highest_rated_genre = (
            genre_df.groupby("genres")["vote_average"]
            .mean()
            .idxmax()
        )

        # ---------------------------------------------------------
        # Most Profitable Budget Range
        # ---------------------------------------------------------

        budget_bins = pd.cut(
            insight_df["budget"],
            bins=5
        )

        best_budget = (
            insight_df.groupby(budget_bins)["ROI"]
            .mean()
            .idxmax()
        )

        # ---------------------------------------------------------
        # Best Runtime Range
        # ---------------------------------------------------------

        runtime_bins = pd.cut(
            insight_df["runtime"],
            bins=5
        )

        best_runtime = (
            insight_df.groupby(runtime_bins)["vote_average"]
            .mean()
            .idxmax()
        )

        # ---------------------------------------------------------
        # Success Percentage
        # ---------------------------------------------------------

        success_percentage = (
            (insight_df["Successful"] == "Successful")
            .mean()
            * 100
        )

        # ---------------------------------------------------------
        # Highest ROI Movie
        # ---------------------------------------------------------

        highest_roi_movie = insight_df.loc[
            insight_df["ROI"].idxmax(),
            "title"
        ]

        # ---------------------------------------------------------
        # Highest Revenue Movie
        # ---------------------------------------------------------

        highest_revenue_movie = insight_df.loc[
            insight_df["revenue"].idxmax(),
            "title"
        ]

        # ---------------------------------------------------------
        # Highest Rated Movie
        # ---------------------------------------------------------

        highest_rated_movie = insight_df.loc[
            insight_df["vote_average"].idxmax(),
            "title"
        ]

        # ---------------------------------------------------------
        # Most Popular Movie
        # ---------------------------------------------------------

        most_popular_movie = insight_df.loc[
            insight_df["popularity"].idxmax(),
            "title"
        ]

        # ---------------------------------------------------------
        # Average Runtime
        # ---------------------------------------------------------

        avg_runtime = insight_df["runtime"].mean()

        # ---------------------------------------------------------
        # Display Insights
        # ---------------------------------------------------------

        st.write(
            f"🎭 **{highest_revenue_genre}** is currently the highest revenue-generating genre in the selected dataset, indicating strong commercial performance compared to other genres."
        )

        st.write(
            f"⭐ Movies in the **{highest_rated_genre}** genre receive the highest average audience ratings, suggesting greater viewer satisfaction."
        )

        st.write(
            f"💰 Movies with budgets between **₹{best_budget.left/1e7:.2f} Cr** and **₹{best_budget.right/1e7:.2f} Cr** deliver the highest average return on investment (ROI)."
        )

        st.write(
            f"⏱ Movies with runtimes between **{best_runtime.left:.0f}** and **{best_runtime.right:.0f} minutes** tend to achieve the best audience ratings."
        )

        st.write(
            f"✅ Approximately **{success_percentage:.2f}%** of the movies in the current selection are classified as commercially successful."
        )

        st.write(
            f"🏆 **{highest_roi_movie}** delivers the highest return on investment, making it the most financially efficient movie in the selected dataset."
        )

        st.write(
            f"🎬 **{highest_revenue_movie}** generated the highest total box office revenue among all filtered movies."
        )

        st.write(
            f"🌟 **{highest_rated_movie}** has the highest audience rating, reflecting exceptional viewer reception."
        )

        st.write(
            f"🔥 **{most_popular_movie}** is the most popular movie based on the popularity score, indicating strong public interest."
        )

        st.write(
            f"⌛ The average runtime of the selected movies is **{avg_runtime:.1f} minutes**, providing an overview of the typical movie length."
        )

    #==================================================================================================================================================================================================================================
    # 📝 Business Recommendations
    #==================================================================================================================================================================================================================================

    if analysis_section == "📝 Business Recommendations":

        st.caption(
            "Actionable recommendations generated from the analysis to improve future movie performance."
        )

        recommendation_df = df.copy()

        recommendation_df["Profit"] = (
            recommendation_df["revenue"] -
            recommendation_df["budget"]
        )

        recommendation_df["ROI"] = np.where(
            recommendation_df["budget"] > 0,
            ((recommendation_df["revenue"] -
            recommendation_df["budget"])
            / recommendation_df["budget"]) * 100,
            0
        )

        recommendation_df["Successful"] = np.where(
            recommendation_df["vote_average"] >= 7,
            1,
            0
        )

        genre_df = recommendation_df.copy()

        genre_df["genres"] = genre_df["genres"].str.split("|")

        genre_df = genre_df.explode("genres")

        # -------------------------------------------------------
        # Generate Recommendations
        # -------------------------------------------------------

        best_genre = (
            genre_df.groupby("genres")["revenue"]
            .mean()
            .idxmax()
        )

        best_runtime = (
            recommendation_df.groupby(
                pd.cut(
                    recommendation_df["runtime"],
                    bins=5
                )
            )["vote_average"]
            .mean()
            .idxmax()
        )

        best_budget = (
            recommendation_df.groupby(
                pd.cut(
                    recommendation_df["budget"],
                    bins=5
                )
            )["ROI"]
            .mean()
            .idxmax()
        )

        success_rate = (
            recommendation_df["Successful"].mean() * 100
        )

        avg_rating = recommendation_df["vote_average"].mean()

        avg_popularity = recommendation_df["popularity"].mean()

        st.markdown("### 📌 Strategic Recommendations")

        st.write(
            f"1️⃣ Prioritize **{best_genre}** movies, as they generate the highest average revenue."
        )

        st.write(
            f"2️⃣ Target a runtime between **{best_runtime.left:.0f}–{best_runtime.right:.0f} minutes**, where movies tend to receive better ratings."
        )

        st.write(
            f"3️⃣ Focus investment within the **₹{best_budget.left/1e7:.2f}–₹{best_budget.right/1e7:.2f} Crore** budget range, which delivers the highest ROI."
        )

        st.write(
            f"4️⃣ Increase marketing efforts for movies expected to exceed the average popularity score of **{avg_popularity:.2f}**."
        )

        st.write(
            f"5️⃣ Aim for an audience rating above **{avg_rating:.2f}**, as highly rated movies generally perform better commercially."
        )

        st.write(
            "6️⃣ Optimize production budgets instead of simply increasing spending, as higher budgets do not always guarantee higher revenue."
        )

        st.write(
            "7️⃣ Prioritize projects with strong ROI potential before approving large production budgets."
        )

        st.write(
            "8️⃣ Use historical performance by genre to guide future investment decisions."
        )

        st.write(
            f"9️⃣ Maintain or improve the current movie success rate of **{success_rate:.2f}%** through data-driven project selection."
        )

        st.write(
            "🔟 Combine audience ratings, popularity, and financial metrics when deciding future movie investments instead of relying on a single metric."
        )