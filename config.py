import os

# Recognition and liveness thresholds
SIMILARITY_THRESHOLD = 0.42
IDENTITY_CONFIDENCE_THRESHOLD = 0.70
LIVENESS_THRESHOLD = 0.7
HEURISTIC_LIVENESS_THRESHOLD = 0.35
ENROLLMENT_LIVENESS_THRESHOLD = 0.30

# Camera source: integer index for local camera or URL string for IP camera/phone stream
CAMERA_SOURCE = 0
FRAME_WIDTH = 480
FRAME_HEIGHT = 352
TARGET_FPS = 30
MAX_FACES_PER_FRAME = 5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
LIVENESS_MODEL_PATH = os.path.join(DATA_DIR, "liveness_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
LOG_PATH = os.path.join(BASE_DIR, "logs", "system.log")

# Enrollment
MIN_ENROLLMENT_IMAGES = 3

# Logging
LOG_LEVEL = "INFO"

# UI/default behavior
API_REFRESH_SECONDS = 5

# Desktop UI tuning
TK_PROCESS_EVERY_N_FRAMES = 4
TK_UI_REFRESH_MS = 16
TK_STATS_REFRESH_MS = 2000

# Blink-based liveness/attendance gating
EYE_AR_CLOSED_THRESHOLD = 0.18
EYE_AR_OPEN_THRESHOLD = 0.23
EYE_AR_ADAPTIVE_ENABLE = False
EYE_AR_BASELINE_ALPHA = 0.05
EYE_AR_CLOSED_RATIO = 0.70
EYE_AR_OPEN_RATIO = 0.82
BLINKS_REQUIRED_FOR_ATTENDANCE = 2
BLINK_MIN_CLOSED_FRAMES = 2
BLINK_SEQUENCE_TIMEOUT_SECONDS = 8
BLINK_TRACK_CLEANUP_SECONDS = 5

# Camera selection dropdown
MAX_LOCAL_CAMERA_INDEX = 5
