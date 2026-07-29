import streamlit as st
import pandas as pd
import time

#===========================================================================================================================================================================================
# Core Notification Engine Setup (Replaces broken JavaScript approach)
#===========================================================================================================================================================================================

def pop(message: str):
    """Call this anywhere to show a message. It automatically stays for at least 3 seconds."""
    
    # 1. If a message was already on screen, make sure it stayed for at least 3 seconds
    if "last_pop_time" in st.session_state:
        elapsed_time = time.perf_counter() - st.session_state.last_pop_time
        remaining_time = 3.0 - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)  # Wait out the rest of the 3 seconds
            
    # 2. Create a clean spot on the screen if it doesn't exist
    if "pop_container" not in st.session_state:
        st.session_state.pop_container = st.empty()
        
    # 3. Save the exact start time of THIS new message
    st.session_state.last_pop_time = time.perf_counter()
    
    # 4. Display the message with a spinning loader
    with st.session_state.pop_container.container():
        st.markdown(
            f"""
            <div style="
                padding: 18px; 
                background-color: #f0f2f6; 
                border-left: 5px solid #ff4b4b; 
                border-radius: 8px;
                margin: 15px 0;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <div style="
                    width: 16px; 
                    height: 16px; 
                    border: 3px solid #ccc; 
                    border-top: 3px solid #ff4b4b; 
                    border-radius: 50%; 
                    animation: spin 1s linear infinite;
                "></div>
                <span style="font-family: sans-serif; font-weight: bold; color: #31333F;">
                    {message}
                </span>
            </div>
            <style>
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            </style>
            """, 
            unsafe_allow_html=True
        )

def close_pop():
    """Call this at the very end of your tasks to make the box disappear."""
    if "last_pop_time" in st.session_state:
        # Make sure the last message gets its full 3 seconds too
        elapsed_time = time.perf_counter() - st.session_state.last_pop_time
        remaining_time = 3.0 - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)
            
    # Clear the screen completely
    if "pop_container" in st.session_state:
        st.session_state.pop_container.empty()
        del st.session_state.pop_container
    if "last_pop_time" in st.session_state:
        del st.session_state.last_pop_time

#===========================================================================================================================================================================================
# Add File Upload box & main heads
#===========================================================================================================================================================================================

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>MOVIE IQ: FILM SUCCESS PREDICTOR</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Analyze and explore your movie dataset instantly.</p>", unsafe_allow_html=True)

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

dataset = st.file_uploader("Upload a CSV file to begin", type=["csv"])

if dataset is not None:
    if not st.session_state.uploaded:
        pop("dataset Uploaded successfully!")
        st.session_state.uploaded = True
        
else:
    # 3. Reset the tracker if the user clears the file
    st.session_state.uploaded = False

#===========================================================================================================================================================================================
# verifying whether dataset has all required columns or not
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
        # Automatically hides the message box once your app moves past initialization steps
        close_pop()

#===========================================================================================================================================================================================
# checking and correcting any missing values in the dataset
#===========================================================================================================================================================================================
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
            close_pop()
            st.rerun()

    # --- CHOICE 2: FILL VALUES MANUALLY OR AUTOMATICALLY ---
    elif action == "2. Fill values (Manually or Automatically)":
        fill_method = st.selectbox("How to fill:", ["Enter a custom value", "Automatic (Mean / Median / Mode)"])
        
        if fill_method == "Enter a custom value":
            custom_val = st.text_input("Type the value to fill into all missing spaces:")
            if st.button("Apply Custom Fill", type="primary"):
                st.session_state.df_working = df.fillna(custom_val)
                pop("Missing data filled!")
                close_pop()
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
                        elif strategy == "Mode (Most Frequent)":
                            updated_df[col] = updated_df[col].fillna(updated_df[col].mode()[0])
                st.session_state.df_working = updated_df
                pop("Automatic calculation complete!")
                close_pop()
                st.rerun()

    # --- CHOICE 3: DELETE ALL ROWS CONTAINING MISSING VALUES ---
    elif action == "3. Delete all rows containing missing values":
        if st.button("Clear All Missing Data Rows", type="primary"):
            st.session_state.df_working = df.dropna()
            pop("All rows containing empty columns removed!")
            close_pop()
            st.rerun()

# Provide a simple trigger to open the cleaning modal inside a Streamlit modal dialog
if "df_working" in st.session_state:
    if st.button("Open Missing Values Cleaner"):
        with st.modal("Fill Missing Values", True):
            clean_data_modal()
