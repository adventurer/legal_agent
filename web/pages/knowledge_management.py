#!/usr/bin/env python3
"""Knowledge library administration page."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import httpx
import streamlit as st

from web.api_client import check_gateway_health
from web.knowledge_ui import render_knowledge_management_page


st.set_page_config(
    page_title="知识库管理 | Legal Agent Lab",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        @media (max-width: 768px) {
            [data-testid="stSidebar"] { display: block !important; }
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 98% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
api_base_url = "http://127.0.0.1:9000"
try:
    health = check_gateway_health(api_base_url)
    gateway_healthy = health.get("status") == "healthy"
except httpx.HTTPError:
    gateway_healthy = False

render_knowledge_management_page(api_base_url, gateway_healthy)
