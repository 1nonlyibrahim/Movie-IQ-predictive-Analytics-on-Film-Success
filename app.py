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

if uploaded_file is not None:
    # Check Required Columns and Missing Values in sidebar
    with st.sidebar:
            required_columns = [
                "budget",
                "revenue",
                "genres",
                "popularity",
                "runtime",
                "vote_average",
            ]
            missing_columns = [
                col for col in required_columns if col not in st.session_state.df.columns
            ]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if len(missing) > 0:
                st.subheader("Missing Values Found")
                st.dataframe(missing)

                # Updated options focusing on ROWS
                action = st.radio(
                "Select how to handle missing data:",
                [
                    "Fill Individually by Cell",
                    "Delete Selected Rows",
                    "Delete All Missing (Rows)",
                ],
            )

            # 1. Fill Individually by Cell
            if action == "Fill Individually by Cell":
                # Find the exact coordinates (Row Index, Column) of missing data
                missing_coordinates = []
                for col in df.columns:
                    null_indices = df[df[col].isnull()].index.tolist()
                    for idx in null_indices:
                        missing_coordinates.append(f"Row {idx} -> Column: {col}")

                # Let user select the exact specific cell
                selected_cell = st.selectbox(
                    "Choose exact missing cell to fill:", missing_coordinates
                )

                if selected_cell:
                    # Extract the row index and column name back out of the string
                    parts = selected_cell.split(" -> Column: ")
                    row_idx = int(parts[0].replace("Row ", ""))
                    col_name = parts[1]

                    # Text input for the replacement value
                    fill_value = st.text_input(f"Value for Row {row_idx}, {col_name}:")

                    if st.button("Apply Cell Fill"):
                        if fill_value:
                            # Handle basic number typing
                            try:
                                fill_value = (
                                    float(fill_value)
                                    if "." in fill_value
                                    else int(fill_value)
                                )
                            except ValueError:
                                pass

                            # Update the single specific cell precisely
                            st.session_state.df.at[row_idx, col_name] = fill_value
                            st.success(f"Updated cell at row {row_idx}!")
                            st.rerun()

            # 2. Delete Selected Rows
            elif action == "Delete Selected Rows":
                # Identify which unique row indexes have at least one NaN
                rows_with_nan = df[df.isnull().any(axis=1)].index.tolist()

                selected_rows = st.multiselect(
                    "Select specific row indices to delete:",
                    options=rows_with_nan,
                    format_func=lambda x: f"Row index {x}",
                )

                if st.button("Delete Selected Rows"):
                    if selected_rows:
                        st.session_state.df = st.session_state.df.drop(
                            index=selected_rows
                        )
                        st.success(f"Deleted rows: {selected_rows}")
                        st.rerun()

            # 3. Delete All Missing (Rows)
            elif action == "Delete All Missing (Rows)":
                st.warning("This deletes any row containing a missing value.")
                if st.button("Confirm Row Deletion"):
                    st.session_state.df = st.session_state.df.dropna()
                    st.success("Successfully dropped rows with missing values!")
                    st.rerun()

            else:
                st.success("✨ No missing values detected in the dataset!")

    # --- Main App Body ---
    st.title("Dataset Dashboard")
    st.write("### Current Data Frame")
    st.dataframe(st.session_state.df)



    # Duplicate Value Detection-----------------------------------------------------------------------------------
    st.header("🔁 Duplicate Value Check")

    duplicate_rows = st.session_state.df[st.session_state.df.duplicated(keep=False)]

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

            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
            removed = before - len(st.session_state.df)

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
                st.session_state.df,
                use_container_width=True,
                num_rows="dynamic"
            )

            st.session_state.df = edited_df

            st.success("Changes applied successfully.")
if uploaded_file is None:
    st.session_state.u = False
    st.info("Please upload a CSV file to begin.")