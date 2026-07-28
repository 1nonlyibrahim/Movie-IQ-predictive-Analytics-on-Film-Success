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
    # store dataframe in session state for editing operations
    st.session_state.df = df.copy()
    if not st.session_state.u:
        pop("✅ Dataset uploaded successfully!")
        st.session_state.u = True

# End of file
st.stop()

import streamlit as st

if uploaded_file is not None:
    df = st.session_state.df

    with st.sidebar:
        st.title("Data Cleaner")
        req = ["budget", "revenue", "genres", "popularity", "runtime", "vote_average"]
        miss_cols = [c for c in req if c not in df.columns]
        if miss_cols:
            st.error(f"Missing columns: {', '.join(miss_cols)}")
            st.stop()

        miss = df.isnull().sum()
        miss = miss[miss > 0]

        if not miss.empty:
            st.subheader("Missing Values Found")
            st.dataframe(miss)
            act = st.radio("Handle missing data:", ["Fill Cell", "Delete Selected Rows", "Delete All Missing Rows"])

            if act == "Fill Cell":
                coords = [f"Row {i} -> Column: {c}" for c in df.columns for i in df[df[c].isnull()].index]
                cell = st.selectbox("Choose missing cell:", coords)
                if cell:
                    r_idx = int(cell.split(" -> Column: ")[0].replace("Row ", ""))
                    c_name = cell.split(" -> Column: ")[1]
                    val = st.text_input(f"Value for Row {r_idx}, {c_name}:")
                    if st.button("Apply Fill") and val:
                        try:
                            val = float(val) if "." in val else int(val)
                        except ValueError:
                            pass
                        st.session_state.df.at[r_idx, c_name] = val
                        st.success(f"Updated row {r_idx}!")
                        st.rerun()

            elif act == "Delete Selected Rows":
                r_nan = df[df.isnull().any(axis=1)].index.tolist()
                sel_r = st.multiselect("Select row indices:", r_nan, format_func=lambda x: f"Row {x}")
                if st.button("Delete Rows") and sel_r:
                    st.session_state.df = df.drop(index=sel_r)
                    st.success(f"Deleted rows: {sel_r}")
                    st.rerun()

            elif act == "Delete All Missing Rows":
                st.warning("Deletes any row with a missing value.")
                if st.button("Confirm Deletion"):
                    st.session_state.df = df.dropna()
                    st.success("Dropped rows!")
                    st.rerun()
        else:
            st.success("✨ No missing values!")

    st.title("Dataset Dashboard")
    st.write("### Current Data Frame")
    st.dataframe(st.session_state.df)

    df = st.session_state.df
    st.header("🔁 Duplicate Value Check")
    dupes = df[df.duplicated(keep=False)]

    if dupes.empty:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠️ {dupes.shape[0]} duplicate rows detected.")
        st.dataframe(dupes, use_container_width=True)
        opt = st.radio("Handle duplicates:", ["Keep All", "Remove All", "Manually Edit"])

        if opt == "Remove All":
            before = len(df)
            st.session_state.df = df.drop_duplicates().reset_index(drop=True)
            st.success(f"✅ {before - len(st.session_state.df)} rows removed.")
            st.rerun()
        elif opt == "Manually Edit":
            st.info("Edit rows manually below.")
            st.session_state.df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        elif opt == "Keep All":
            st.info("Duplicate rows kept.")

else:
    st.session_state.u = False
    st.info("Please upload a CSV file to begin.")