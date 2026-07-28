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

st.markdown("""
<style>
/* Full screen container for the loading animation */
.full-screen-loader {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(255, 255, 255, 0.85);
    z-index: 999998;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
}
/* Spinner ring */
.spinner-ring {
    border: 6px solid #f3f3f3;
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
</style>
""", unsafe_allow_html=True)

# 3. Your Custom Top-Middle Popup Function
def pop(m):
    st.markdown(
        f'<div style="position:fixed;left:50%;transform:translateX(-50%);background:#d4edda;color:#155724;padding:14px 28px;border-radius:8px;font-family:sans-serif;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.15);z-index:999999;width:max-content;animation:s 3s cubic-bezier(0.25,1,0.5,1) forwards">{m}</div>'
        f'<style>@keyframes s{{0%{{top:-100px;opacity:0}}15%{{top:30px;opacity:1}}85%{{top:30px;opacity:1}}100%{{top:-100px;opacity:0}}}}</style>', 
        unsafe_allow_html=True
    )

# 4. Initialize State Flags
if "u" not in st.session_state: 
    st.session_state.u = False      # Track file upload status
if "trigger_pop" not in st.session_state: 
    st.session_state.trigger_pop = False  # Track popup timing

# 5. File Uploader Interface
dataset = st.file_uploader("Upload your movie dataset", type=["csv"])

# 6. Logic Flow
if dataset is not None and not st.session_state.u:
    # A. Display the middle-of-the-screen loading container
    loader_placeholder = st.empty()
    with loader_placeholder:
        st.markdown("""
            <div class="full-screen-loader">
                <div class="spinner-ring"></div>
                <div class="loader-text">Uploading movie dataset...</div>
            </div>
        """, unsafe_allow_html=True)
        
        # B. Simulate processing time for the file
        time.sleep(2.5) 
    
    # C. Clear loader, update states, and rerun to transition smoothly
    loader_placeholder.empty()
    st.session_state.u = True
    st.session_state.trigger_pop = True
    st.rerun()

# 7. Render Success Popup (Only after loader finishes)
if st.session_state.trigger_pop:
    pop("Dataset uploaded successfully!")
    st.session_state.trigger_pop = False # Clear flag so it doesn't loop infinitely