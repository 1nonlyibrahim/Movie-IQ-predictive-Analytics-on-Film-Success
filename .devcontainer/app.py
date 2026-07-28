import streamlit as st
import pandas as pd

#Add File Upload---------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.title("Movie IQ: Film Success Predictor")

uploaded_file = st.file_uploader(
    "Upload your movie dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

#Show Dataset Shape------------------------------------------------------------------------------------------

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

col1, col2 = st.columns(2)

col1.metric("Rows", df.shape[0])

col2.metric("Columns", df.shape[1])

#Display Dataset Information-----------------------------------------------------------------------------

st.subheader("Column Information")

info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(info)

#Check Required Columns--------------------------------------------------------------------------------------

required_columns = [
    "budget",
    "revenue",
    "genres",
    "popularity",
    "runtime",
    "vote_average"
]
missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]
if missing_columns:

    st.error(
        f"Missing required columns: {', '.join(missing_columns)}"
    )

    st.stop()

else:

    st.success("All required columns are available.")

#Check Missing Values----------------------------------------------------------------------------------------
missing = df.isnull().sum()

missing = missing[missing > 0]

st.subheader("Missing Values")
if len(missing) == 0:

    st.success("No missing values found.")

else:

    st.dataframe(missing)