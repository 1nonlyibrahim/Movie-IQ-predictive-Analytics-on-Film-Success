import streamlit as st
import pandas as pd
import time


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

dataset = st.file_uploader("Upload your movie dataset", type=["csv"])

# Initialize Session State Variables Safely
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False      
if "trigger_pop" not in st.session_state:
    st.session_state.trigger_pop = False  

# 2. Re-inject Global CSS for Full-Screen Loader and Popup Animations
st.markdown("""
<style>
/* Full screen overlay container for the middle-screen spinner */
.full-screen-loader {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(255, 255, 255, 0.9);
    z-index: 999998;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
}
/* Spinner ring */
.spinner-ring {
    border: 6px solid #e0e0e0;
    border-top: 6px solid #28a745;
    border-radius: 50%;
    width: 60px; height: 60px;
    animation: spin 1s linear infinite;
}
@keyframes spin { 
    0% { transform: rotate(0deg); } 
    100% { transform: rotate(360deg); } 
}
.loader-text {
    margin-top: 15px; font-family: sans-serif;
    font-weight: bold; color: #333; font-size: 18px;
}
/* Slide down-up popup style */
@keyframes slide-popup {
    0%   { top: -100px; opacity: 0; }
    15%  { top: 30px; opacity: 1; }
    85%  { top: 30px; opacity: 1; }
    100% { top: -100px; opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# 3. Your Top-Middle Success Popup Function
def pop(message):
    """Renders the custom green alert box at the top center of the page."""
    st.markdown(
        f'<div style="'
        f'position: fixed; left: 50%; transform: translateX(-50%); '
        f'background: #d4edda; color: #155724; padding: 14px 28px; '
        f'border-radius: 8px; font-family: sans-serif; font-weight: bold; '
        f'box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 999999; '
        f'width: max-content; animation: slide-popup 3s cubic-bezier(0.25, 1, 0.5, 1) forwards;">'
        f'{message}'
        f'</div>', 
        unsafe_allow_html=True
    )

st.title("Movie IQ Analytics")

# 4. File Uploader Interface 
# FIXED: Assigned unique static key 'movie_csv_uploader' to block Duplicate Element registration errors
dataset = st.file_uploader(
    "Upload your movie dataset", 
    type=["csv"], 
    key="movie_csv_uploader"
)

# 5. Application State Execution Flow
if dataset is not None and not st.session_state.uploaded:
    # Set up container block for loading phase
    loader_placeholder = st.empty()
    with loader_placeholder:
        st.markdown("""
            <div class="full-screen-loader">
                <div class="spinner-ring"></div>
                <div class="loader-text">Uploading movie dataset...</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(2.5) # Simulates processing sequence
    
    # Tear down loader container, flag state markers, and enforce clean page redraw
    loader_placeholder.empty()
    st.session_state.uploaded = True
    st.session_state.trigger_pop = True
    st.rerun()

# 6. Fire Success Popup sequence safely
if st.session_state.trigger_pop:
    pop("Dataset uploaded successfully!")
    st.session_state.trigger_pop = False

# Reset flags cleanly if the user clears out the file uploader interface later
if dataset is None and st.session_state.uploaded:
    st.session_state.uploaded = False