#!/usr/bin/env python3
"""Streamlit multipage entrypoint."""

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR.parent))

import streamlit as st


pages = [
    st.Page("app_ui.py", title="合同审查", icon="⚖️", default=True),
    st.Page("pages/knowledge_management.py", title="知识库管理", icon="📚"),
]
st.navigation(pages, position="top").run()
