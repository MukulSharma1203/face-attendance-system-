# Face Attendance System

This is the current desktop version of the project.

It uses:
- Tkinter for the main app UI
- InsightFace for face detection + embeddings
- MediaPipe/landmark EAR for blink confirmation
- SQLite for records
- CSV exports per date

## What It Does

- Enrolls users directly from live camera capture
- Recognizes faces in real time
- Marks attendance only after blink confirmation
- Prevents duplicate attendance on the same day
- Stores attendance in `attendance.db`
- Exports date-wise CSV files in `data/processed/`

## Current Project Layout

- `desktop_app.py` - main attendance app (run this first)
- `attendance_viewer.py` - desktop viewer for date-wise attendance
- `attendance_web.py` - browser viewer (Streamlit)
- `embeddings.py` - InsightFace detection and embedding extraction
- `liveness.py` - liveness helper logic
- `database.py` - SQLite + CSV utilities
- `utils.py` - logging, similarity, and helpers
- `config.py` - all thresholds and paths
- `attendance.db` - local database
- `data/raw/` - enrollment captures
- `data/processed/` - attendance CSVs (`attendance_YYYY-MM-DD.csv`)
- `models/` - MediaPipe model files (downloaded when needed)
- `logs/` - runtime logs

## Install

```bash
py -3.11 -m pip install -r requirements.txt
```

## Run

Main app:

```bash
py -3.11 desktop_app.py
```

Desktop viewer:

```bash
py -3.11 attendance_viewer.py
```

Web viewer:

```bash
py -3.11 -m streamlit run attendance_web.py
```

## Typical Flow

1. Run `desktop_app.py`.
2. Select camera from dropdown and switch if needed.
3. Add a user with auto-capture enrollment.
4. During recognition, blink as prompted to confirm attendance.
5. Open desktop/web viewer to check records by date.

## Notes

- Identity confidence below the configured threshold is treated as `Unknown/Spoof`.
- If MediaPipe face landmarker is not available, the app falls back to available landmark path.
- Date dropdown in viewers reads both DB dates and existing CSV dates.
