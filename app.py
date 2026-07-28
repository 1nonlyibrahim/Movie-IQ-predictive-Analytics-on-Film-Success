import streamlit as st
import pandas as pd


#Add File Upload---------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>MOVIE IQ: FILM SUCCESS PREDICTOR</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Analyze and explore your movie dataset instantly.</p>", unsafe_allow_html=True)

def pop(m):
    st.markdown(f'<div style="position:fixed;left:50%;transform:translateX(-50%);background:#d4edda;color:#155724;padding:14px 28px;border-radius:8px;font-family:sans-serif;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.15);z-index:999999;width:max-content;animation:s 3s cubic-bezier(0.25,1,0.5,1) forwards">{m}</div><style>@keyframes s{{0%{{top:-100px;opacity:0}}15%{{top:30px;opacity:1}}85%{{top:30px;opacity:1}}100%{{top:-100px;opacity:0}}}}</style>', unsafe_allow_html=True)

if "u" not in st.session_state: st.session_state.u = False

uploaded_file = st.file_uploader("Upload your movie dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not st.session_state.u:
        pop("✅ Dataset uploaded successfully!")
        st.session_state.u = True

else:
    st.session_state.u = False
    st.info("Please upload a CSV file to begin.")
    st.stop()

if uploaded_file is not None:
    #Check Required Columns--------------------------------------------------------------------------------------
    with st.sidebar:
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

        if len(missing) > 0:
            st.subheader("Missing Values")
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
else:
    st.session_state.u = False
    st.info("Please upload a CSV file to begin.")