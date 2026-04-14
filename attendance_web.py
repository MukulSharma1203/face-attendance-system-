import os
from datetime import datetime

import pandas as pd
import streamlit as st

import config
import database

st.set_page_config(page_title="Attendance Web Viewer", page_icon="📋", layout="wide")

st.markdown("## Attendance Web Viewer")
st.caption("Date-wise attendance from CSV files and database records")


def get_date_options():
    database.init_db()
    return database.get_all_available_attendance_dates()


def load_rows_for_date(date_str: str):
    rows = database.get_attendance_for_date(date_str)
    if rows:
        return rows
    return database.load_attendance_csv_by_date(date_str)


def to_df(rows):
    if not rows:
        return pd.DataFrame(columns=["name", "date", "time", "status"])
    df = pd.DataFrame(rows)
    cols = [c for c in ["name", "date", "time", "status"] if c in df.columns]
    return df[cols]


dates = get_date_options()

if not dates:
    st.warning("No attendance files or DB records found yet.")
    st.stop()

selected_date = st.sidebar.selectbox("Select date", dates, index=0)

st.subheader(f"Attendance for {selected_date}")
rows = load_rows_for_date(selected_date)
df = to_df(rows)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Marked", int(len(df)))
with col2:
    st.metric("Last Refreshed", datetime.now().strftime("%H:%M:%S"))

st.dataframe(df, use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"attendance_{selected_date}.csv",
    mime="text/csv",
)

if st.button("Refresh"):
    st.rerun()
