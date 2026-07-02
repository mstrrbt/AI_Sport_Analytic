#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app/coachmind_app.py
