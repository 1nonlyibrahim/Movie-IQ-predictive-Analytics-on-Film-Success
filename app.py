import streamlit as st
import pandas as pd
import time

#===========================================================================================================================================================================================
#Add File Upload box & main heads
#===========================================================================================================================================================================================

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>MOVIE IQ: FILM SUCCESS PREDICTOR</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Analyze and explore your movie dataset instantly.</p>", unsafe_allow_html=True)

def pop(m):
    uid = int(time.time() * 1000000) + hash(m) % 1000
    st.markdown(
        f"""
        <div id="notif-trigger-{uid}" style="display:none;"></div>
        <script>
        (function() {{
            let container = document.getElementById('notif-stack-container');
            if (!container) {{
                container = document.createElement('div');
                container.id = 'notif-stack-container';
                container.style.position = 'fixed';
                container.style.top = '30px';
                container.style.left = '50%';
                container.style.transform = 'translateX(-50%)';
                container.style.zIndex = '999999';
                container.style.display = 'flex';
                container.style.flexDirection = 'column';
                container.style.gap = '10px';
                container.style.alignItems = 'center';
                container.style.width = 'max-content';
                container.style.pointerEvents = 'none';
                document.body.appendChild(container);
            }}

            const el = document.createElement('div');
            el.style.background = '#d4edda';
            el.style.color = '#155724';
            el.style.padding = '14px 28px';
            el.style.borderRadius = '8px';
            el.style.fontFamily = 'sans-serif';
            el.style.fontWeight = 'bold';
            el.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            el.style.width = 'max-content';
            el.style.pointerEvents = 'auto';
            el.style.animation = 'slideInDown 3s cubic-bezier(0.25, 1, 0.5, 1) forwards';

            el.innerHTML = `<span>{m}</span>`;

            if (!document.getElementById('notif-stack-styles')) {{
                const style = document.createElement('style');
                style.id = 'notif-stack-styles';
                style.innerHTML = `
                    @keyframes slideInDown {{
                        0% {{ transform: translateY(-50px); opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; margin-bottom: 0; overflow: hidden; }}
                        15% {{ transform: translateY(0); opacity: 1; max-height: 100px; }}
                        85% {{ transform: translateY(0); opacity: 1; max-height: 100px; opacity: 1; }}
                        100% {{ transform: translateY(-50px); opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; margin-bottom: 0; overflow: hidden; }}
                    }}
                `;
                document.head.appendChild(style);
            }}

            container.appendChild(el);

            setTimeout(() => {{
                el.remove();
                if (container.childElementCount === 0) {{
                    container.remove();
                }}
            }}, 3000);
        }})();
        </script>
        """,
        unsafe_allow_html=True
    )
    
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

dataset = st.file_uploader("Upload a CSV file to begin", type=["csv"])

if dataset is not None:
    if not st.session_state.uploaded:
        pop("dataset Uploaded successfully!")
        
else:
    # 3. Reset the tracker if the user clears the file
    st.session_state.uploaded = False

#===========================================================================================================================================================================================
#verifying wether dataset has all required columns or not
#===========================================================================================================================================================================================

# --- Quick Gatekeeper Check for Specific Columns ---
if dataset is not None:
    # Initialize session state if not present
    if "df_working" not in st.session_state:
        st.session_state.df_working = pd.read_csv(dataset)

    df_current = st.session_state.df_working
    
    # Define your specific target columns
    required_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'title', 'genres']
    
    # Ensure all target columns actually exist in the dataframe before checking
    existing_cols = [col for col in required_columns if col in df_current.columns]
    
    # Quick check: returns True if ANY cell in these specific columns is missing
    is_dataset_incomplete = df_current[existing_cols].isnull().any().any()

    if is_dataset_incomplete:
        st.error("🛑 Analysis Halted: Required movie data columns contain missing values.")
        st.info("Please fill or remove missing data in your core fields (budget, revenue, runtime, vote_average, title, genres) to proceed.")
        
        # Force stop execution here to prevent further analysis charts/tables from rendering
        st.stop()
    else:
        pop("✅ Movie dataset columns are complete! Proceeding to analysis...")


#===========================================================================================================================================================================================
#checking and correcting any missing values in the dataset
#===========================================================================================================================================================================================
@st.dialog("Fill Missing Values", width="large")
def clean_data_modal():
    # Fetch the dataset from session state
    df = st.session_state.df_working
    missing_rows = df[df.isnull().any(axis=1)]
    
    st.write("Below are all the rows that contain at least one missing column:")
    # Task 1: Show complete rows with any missing columns
    st.dataframe(missing_rows)
    
    st.divider()
    
    # Task 2: Provide the 3 choices
    action = st.radio(
        "Choose an action:",
        [
            "1. Select and delete specific rows manually",
            "2. Fill values (Manually or Automatically)",
            "3. Delete all rows containing missing values"
        ]
    )
    
    # --- CHOICE 1: SELECT AND DELETE ROWS MANUALLY ---
    if action == "1. Select and delete specific rows manually":
        rows_to_delete = st.multiselect(
            "Select row indices to completely remove:",
            options=missing_rows.index.tolist()
        )
        if st.button("Delete Selected Rows", type="primary"):
            st.session_state.df_working = df.drop(index=rows_to_delete)
            pop("Rows deleted successfully!")
            st.rerun()

    # --- CHOICE 2: FILL VALUES MANUALLY OR AUTOMATICALLY ---
    elif action == "2. Fill values (Manually or Automatically)":
        fill_method = st.selectbox("How to fill:", ["Enter a custom value", "Automatic (Mean / Median / Mode)"])
        
        if fill_method == "Enter a custom value":
            custom_val = st.text_input("Type the value to fill into all missing spaces:")
            if st.button("Apply Custom Fill", type="primary"):
                st.session_state.df_working = df.fillna(custom_val)
                pop("Missing data filled!")
                st.rerun()
                
        elif fill_method == "Automatic (Mean / Median / Mode)":
            strategy = st.selectbox("Select strategy:", ["Mean (Average)", "Median (Middle)", "Mode (Most Frequent)"])
            if st.button("Apply Automatic Fill", type="primary"):
                updated_df = df.copy()
                for col in updated_df.columns:
                    if updated_df[col].isnull().any():
                        if strategy == "Mean (Average)" and pd.api.types.is_numeric_dtype(updated_df[col]):
                            updated_df[col] = updated_df[col].fillna(updated_df[col].mean())
                        elif strategy == "Median (Middle)" and pd.api.types.is_numeric_dtype(updated_df[col]):
                            updated_df[col] = updated_df[col].fillna(updated_df[col].median())
                        else:
                            updated_df[col] = updated_df[col].fillna(updated_df[col].mode().iloc[0] if not updated_df[col].mode().empty else "Missing")
                st.session_state.df_working = updated_df
                pop("Missing data auto-filled!")
                st.rerun()

    # --- CHOICE 3: DELETE ALL THE ROWS ---
    elif action == "3. Delete all rows containing missing values":
        st.warning("This will completely drop all rows with any missing values.")
        if st.button("Confirm Row Deletion", type="primary"):
            st.session_state.df_working = df.dropna()
            pop("Rows dropped successfully!")
            st.rerun()


# --- Main App Execution Logic ---
if dataset is not None:
    # Read the data once and save it to the session state
    if "df_working" not in st.session_state:
        st.session_state.df_working = pd.read_csv(dataset)
        st.session_state.modal_triggered = False

    df_current = st.session_state.df_working
    has_missing = df_current.isnull().any().any()

    # If missing values exist and we haven't resolved them yet
    if has_missing:
        # Show a warning button to open/reopen the modal
        st.error("⚠️ This dataset contains missing values.")
        if st.button("🔧 Open Data Filling Window") or not st.session_state.modal_triggered:
            st.session_state.modal_triggered = True
            clean_data_modal()
    else:
        pop("🎉 Dataset is Ready! No missing values remaining.")

else:
    # Clear session data if the file is removed
    if "df_working" in st.session_state:
        del st.session_state.df_working
        st.session_state.modal_triggered = False

#===========================================================================================================================================================================================
#checking and correcting any duplicates values in the dataset
#===========================================================================================================================================================================================

@st.dialog("Handle Duplicate Rows", width="large")
def handle_duplicates_modal():
    # Fetch the dataset from session state
    df = st.session_state.df_working
    duplicate_rows = df[df.duplicated(keep=False)]
    
    st.write("Below are all the rows that have identical duplicates in the dataset:")
    # Task 1: Show complete rows with any duplicates
    st.dataframe(duplicate_rows)
    
    st.divider()
    
    # Task 2: Provide the 3 choices
    action = st.radio(
        "Choose an action:",
        [
            "1. Select and delete specific duplicate rows manually",
            "2. Keep specific instances (First or Last)",
            "3. Delete all duplicate rows entirely"
        ]
    )
    
    # --- CHOICE 1: SELECT AND DELETE ROWS MANUALLY ---
    if action == "1. Select and delete specific duplicate rows manually":
        rows_to_delete = st.multiselect(
            "Select row indices to completely remove:",
            options=duplicate_rows.index.tolist()
        )
        if st.button("Delete Selected Rows", type="primary"):
            st.session_state.df_working = df.drop(index=rows_to_delete)
            pop("Rows deleted successfully!")
            st.rerun()

    # --- CHOICE 2: KEEP SPECIFIC INSTANCES ---
    elif action == "2. Keep specific instances (First or Last)":
        keep_strategy = st.selectbox("Which instance do you want to keep?", ["Keep First occurrence", "Keep Last occurrence"])
        
        if st.button("Apply Keep Strategy", type="primary"):
            strategy_val = "first" if keep_strategy == "Keep First occurrence" else "last"
            st.session_state.df_working = df.drop_duplicates(keep=strategy_val)
            pop("Duplicates dropped, keeping chosen occurrences!")
            st.rerun()

    # --- CHOICE 3: DELETE ALL THE ROWS ---
    elif action == "3. Delete all duplicate rows entirely":
        st.warning("This will completely drop ALL instances of duplicate rows, leaving none behind.")
        if st.button("Confirm Complete Deletion", type="primary"):
            st.session_state.df_working = df.drop_duplicates(keep=False)
            pop("All duplicate instances dropped successfully!")
            st.rerun()


# --- Main App Execution Logic for Duplicates ---
if dataset is not None:
    # Read the data once and save it to the session state (already done in your main file)
    if "df_working" not in st.session_state:
        st.session_state.df_working = pd.read_csv(dataset)
        st.session_state.duplicate_modal_triggered = False

    df_current = st.session_state.df_working
    has_duplicates = df_current.duplicated().any()

    # If duplicate values exist and we haven't resolved them yet
    if has_duplicates:
        # Show a warning button to open/reopen the modal
        st.error("⚠️ This dataset contains duplicate rows.")
        if st.button("🔧 Open Duplicate Handling Window") or not st.session_state.get('duplicate_modal_triggered', False):
            st.session_state.duplicate_modal_triggered = True
            handle_duplicates_modal()
    else:
        pop("🎉 Dataset is Clean! No duplicate rows remaining.")

    # Always show the current state of the dataset on the main page
    st.subheader("📊 Dataset Preview")
    st.dataframe(df_current)