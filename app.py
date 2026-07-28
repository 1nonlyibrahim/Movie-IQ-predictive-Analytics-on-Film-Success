import streamlit as st
import pandas as pd
import time


#Add File Upload---------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>MOVIE IQ: FILM SUCCESS PREDICTOR</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Analyze and explore your movie dataset instantly.</p>", unsafe_allow_html=True)

def pop(m):
    st.markdown(f'<div style="position:fixed;left:50%;transform:translateX(-50%);background:#d4edda;color:#155724;padding:14px 28px;border-radius:8px;font-family:sans-serif;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.15);z-index:999999;width:max-content;animation:s 3s cubic-bezier(0.25,1,0.5,1) forwards">{m}</div><style>@keyframes s{{0%{{top:-100px;opacity:0}}15%{{top:30px;opacity:1}}85%{{top:30px;opacity:1}}100%{{top:-100px;opacity:0}}}}</style>', unsafe_allow_html=True)

if "u" not in st.session_state: st.session_state.u = False

dataset = st.file_uploader("Upload your movie dataset", type=["csv"])