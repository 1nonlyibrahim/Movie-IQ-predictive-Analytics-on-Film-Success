import streamlit as st
import pandas as pd


#Add File Upload---------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>Movie IQ: Film Success Predictor</h1>", unsafe_allow_html=True)

st.markdown("Analyze and explore your movie dataset instantly.")

uploaded_file = st.file_uploader(
    "Upload your movie dataset",
    type=["csv"]
    )

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    def pop(msg):
        st.components.v1.html(f'<div id="t" style="position:fixed;left:50%;top:-50px;transform:translateX(-50%);background:#d4edda;color:#155724;padding:12px 24px;border-radius:8px;font-family:sans-serif;font-weight:bold;box-shadow:0 4px 6px #0002;transition:all .5s">{msg}</div><script>t=document.getElementById("t");setTimeout(()=>t.style.top="20px",50);setTimeout(()=>t.style.top="-100px",3000)</script>', height=0)
        pop("✅ Dataset uploaded successfully!")



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


#Check Missing Values----------------------------------------------------------------------------------------
missing = df.isnull().sum()

missing = missing[missing > 0]

st.subheader("Missing Values")
if len(missing) > 0:

    st.dataframe(missing)



# Duplicate Value Detection-----------------------------------------------------------------------------------
st.header("🔁 Duplicate Value Check")

duplicate_rows = df[df.duplicated(keep=False)]

if duplicate_rows.empty:
    st.success("✅ No duplicate rows found in the dataset.")

else:
    st.warning(f"⚠️ {duplicate_rows.shape[0]} duplicate rows detected.")

    st.subheader("Duplicate Records")

    st.dataframe(duplicate_rows, use_container_width=True)

    option = st.radio(
        "How would you like to handle duplicate values?",
        (
            "Keep All Duplicates",
            "Remove All Duplicates",
            "Manually Edit Dataset"
        )
    )

    # ----------------------------------------
    # Keep duplicates
    # ----------------------------------------
    if option == "Keep All Duplicates":

        st.info("Duplicate rows have been kept in the dataset.")

    # ----------------------------------------
    # Remove duplicates
    # ----------------------------------------
    elif option == "Remove All Duplicates":

        before = len(df)

        df = df.drop_duplicates().reset_index(drop=True)

        removed = before - len(df)

        st.success(f"✅ {removed} duplicate rows removed successfully.")

    # ----------------------------------------
    # Manual editing
    # ----------------------------------------
    elif option == "Manually Edit Dataset":

        st.info(
            "You can edit or delete rows manually below. "
            "Click any cell to edit it or remove unwanted rows."
        )

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic"
        )

        df = edited_df

        st.success("Changes applied successfully.")