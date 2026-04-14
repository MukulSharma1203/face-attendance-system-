import logging
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from tkinter import messagebox, ttk
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import mediapipe as mp
except Exception:
    mp = None

import config
import database
import embeddings
import liveness
import utils

utils.setup_logging()
logger = logging.getLogger("desktop")


class DesktopAttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition Attendance - Desktop")
        self.root.geometry("1200x760")
        self.root.minsize(1100, 700)

        database.init_db()

        self.capture = None
        self.capture_lock = threading.Lock()
        self.current_source = 0
        self.available_cameras = []

        self.last_frame = None
        self.frame_count = 0

        self.emb_cache: Dict[int, np.ndarray] = {}
        self.emb_mtime = -1.0

        self.blink_state: Dict[int, Dict] = {}
        self.use_mediapipe_blink = False
        self.mediapipe_mode: Optional[str] = None
        self.mp_face_mesh = None
        self.mp_face_landmarker = None
        self.mp_image_cls = None
        self.mp_image_format = None

        self.is_running = False
        self.is_enrolling = False

        self.enroll_window: Optional[tk.Toplevel] = None
        self.enroll_progress_var = tk.DoubleVar(value=0.0)
        self.enroll_status_var = tk.StringVar(value="")
        self.enroll_tip_var = tk.StringVar(value="")

        self._configure_style()
        self._init_blink_engine()
        self._build_ui()
        self.refresh_camera_list()
        if self.available_cameras:
            self.current_source = self.available_cameras[0]
            self.camera_var.set(str(self.current_source))
        self.start_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(config.TK_UI_REFRESH_MS, self.video_loop)
        self.root.after(config.TK_STATS_REFRESH_MS, self.refresh_dashboard)

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f6fb")
        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=1)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#1d3557")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#0b3d91", background="#f4f6fb")
        style.configure("SubTitle.TLabel", font=("Segoe UI", 10), foreground="#556070", background="#f4f6fb")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))

    def _init_blink_engine(self):
        if mp is None:
            logger.warning("MediaPipe not found. Blink detection will use InsightFace landmark fallback.")
            self.use_mediapipe_blink = False
            return

        # Path A: older/newer builds exposing mp.solutions.face_mesh
        if hasattr(mp, "solutions"):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.use_mediapipe_blink = True
                self.mediapipe_mode = "solutions"
                logger.info("MediaPipe solutions.FaceMesh enabled for blink detection")
                return
            except Exception as exc:
                logger.warning("MediaPipe solutions blink init failed (%s).", exc)

        # Path B: tasks-only builds (your environment: mediapipe 0.10.33)
        try:
            tasks_python = __import__("mediapipe.tasks.python", fromlist=["python"])
            tasks_vision = __import__("mediapipe.tasks.python.vision", fromlist=["vision"])

            model_path = self._get_face_landmarker_model_path()
            base_options = tasks_python.BaseOptions(model_asset_path=model_path)
            options = tasks_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=tasks_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self.mp_face_landmarker = tasks_vision.FaceLandmarker.create_from_options(options)
            self.mp_image_cls = mp.Image
            self.mp_image_format = mp.ImageFormat

            self.use_mediapipe_blink = True
            self.mediapipe_mode = "tasks"
            logger.info("MediaPipe tasks.FaceLandmarker enabled for blink detection")
        except Exception as exc:
            logger.warning("MediaPipe blink init failed (%s). Using InsightFace fallback.", exc)
            self.use_mediapipe_blink = False
            self.mediapipe_mode = None

    def _get_face_landmarker_model_path(self) -> str:
        model_dir = os.path.join(config.BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        if os.path.exists(model_path):
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        logger.info("Downloading MediaPipe face_landmarker.task...")
        urllib.request.urlretrieve(model_url, model_path)
        return model_path

    def _build_ui(self):
        self.root.configure(background="#f4f6fb")

        header = ttk.Frame(self.root, padding=(14, 10), style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Face Recognition Attendance", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Desktop mode | Adaptive blink confirmation | Multi-camera dropdown",
            style="SubTitle.TLabel",
        ).pack(anchor="w")

        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        left = ttk.Frame(self.root, padding=10, style="App.TFrame")
        right = ttk.Frame(self.root, padding=10, style="App.TFrame")
        left.grid(row=1, column=0, sticky="nsew")
        right.grid(row=1, column=1, sticky="nsew")

        video_card = ttk.LabelFrame(left, text="Live Camera", padding=8, style="Card.TLabelframe")
        video_card.pack(fill="both", expand=True)

        self.video_label = ttk.Label(video_card)
        self.video_label.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Initializing camera...")
        self.status_label = tk.Label(
            video_card,
            textvariable=self.status_var,
            fg="#0b8f3e",
            bg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(6, 0))

        control_card = ttk.LabelFrame(right, text="Controls", padding=10, style="Card.TLabelframe")
        control_card.pack(fill="x")

        row0 = ttk.Frame(control_card)
        row0.grid(row=0, column=0, sticky="we")
        ttk.Button(row0, text="Restart Camera", command=self.restart_camera, style="Primary.TButton").pack(side="left")

        ttk.Separator(control_card).grid(row=1, column=0, sticky="we", pady=10)

        ttk.Label(control_card, text="Camera Device").grid(row=2, column=0, sticky="w")
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(control_card, textvariable=self.camera_var, state="readonly", width=22)
        self.camera_combo.grid(row=3, column=0, sticky="we", pady=(4, 8))

        row_cam = ttk.Frame(control_card)
        row_cam.grid(row=4, column=0, sticky="we")
        ttk.Button(row_cam, text="Refresh Camera List", command=self.refresh_camera_list, style="Primary.TButton").pack(side="left")
        ttk.Button(row_cam, text="Switch Camera", command=self.switch_camera, style="Primary.TButton").pack(side="left", padx=(8, 0))

        ttk.Separator(control_card).grid(row=5, column=0, sticky="we", pady=10)

        ttk.Label(control_card, text="New User Name").grid(row=6, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(control_card, textvariable=self.name_var, width=42).grid(row=7, column=0, sticky="we", pady=(4, 8))

        ttk.Label(
            control_card,
            text=f"Tip: blink naturally twice to mark attendance. Enrollment threshold {config.ENROLLMENT_LIVENESS_THRESHOLD:.2f}",
            style="SubTitle.TLabel",
        ).grid(row=8, column=0, sticky="w", pady=(0, 8))

        self.enroll_btn = ttk.Button(
            control_card,
            text=f"Auto-Capture {config.MIN_ENROLLMENT_IMAGES} Samples & Register",
            command=self.enroll_user,
            style="Primary.TButton",
        )
        self.enroll_btn.grid(row=9, column=0, sticky="we")

        ttk.Button(
            control_card,
            text="Show Attendance Viewer",
            command=self.open_attendance_viewer,
            style="Primary.TButton",
        ).grid(row=10, column=0, sticky="we", pady=(8, 0))

        control_card.columnconfigure(0, weight=1)

        stats_card = ttk.LabelFrame(right, text="Stats", padding=10, style="Card.TLabelframe")
        stats_card.pack(fill="x", pady=(10, 0))

        self.total_var = tk.StringVar(value="0")
        self.present_var = tk.StringVar(value="0")
        self.absent_var = tk.StringVar(value="0")

        ttk.Label(stats_card, text="Total Registered:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_card, textvariable=self.total_var).grid(row=0, column=1, sticky="e")

        ttk.Label(stats_card, text="Present Today:").grid(row=1, column=0, sticky="w")
        ttk.Label(stats_card, textvariable=self.present_var).grid(row=1, column=1, sticky="e")

        ttk.Label(stats_card, text="Absent Today:").grid(row=2, column=0, sticky="w")
        ttk.Label(stats_card, textvariable=self.absent_var).grid(row=2, column=1, sticky="e")

        stats_card.columnconfigure(1, weight=1)

        table_card = ttk.LabelFrame(right, text="Today Attendance", padding=10, style="Card.TLabelframe")
        table_card.pack(fill="both", expand=True, pady=(10, 0))

        self.table = ttk.Treeview(table_card, columns=("name", "time", "status"), show="headings", height=14)
        self.table.heading("name", text="Name")
        self.table.heading("time", text="Time")
        self.table.heading("status", text="Status")
        self.table.column("name", width=150)
        self.table.column("time", width=110)
        self.table.column("status", width=100)
        self.table.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

    def _open_capture(self, source: int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def scan_local_cameras(self):
        indexes = []
        for idx in range(config.MAX_LOCAL_CAMERA_INDEX + 1):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            ok = bool(cap and cap.isOpened())
            if ok:
                ok, _ = cap.read()
            if ok:
                indexes.append(idx)
            if cap is not None:
                cap.release()
        return indexes

    def refresh_camera_list(self):
        current = int(self.current_source)
        cams = self.scan_local_cameras()
        if not cams:
            cams = [0]
        if current not in cams:
            cams = [current, *cams]

        cams = sorted(set(cams))
        self.available_cameras = cams
        self.camera_combo["values"] = [str(i) for i in cams]
        self.camera_var.set(str(current))

    def start_camera(self):
        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = self._open_capture(int(self.current_source))
            self.is_running = bool(self.capture and self.capture.isOpened())

        if self.is_running:
            self._set_status(f"Camera active: {self.current_source}", ok=True)
        else:
            self._set_status(f"Camera unavailable: {self.current_source}", ok=False)

    def switch_camera(self):
        raw = self.camera_var.get().strip()
        if not raw.isdigit():
            messagebox.showwarning("Camera", "Please select a valid camera index.")
            return

        new_source = int(raw)
        with self.capture_lock:
            test_cap = self._open_capture(new_source)
            ok = bool(test_cap and test_cap.isOpened())
            if ok:
                ok, _ = test_cap.read()

            if not ok:
                if test_cap is not None:
                    test_cap.release()
                self._set_status(f"Failed to open camera: {new_source}", ok=False)
                messagebox.showerror("Camera", f"Could not open camera index {new_source}.")
                return

            if self.capture is not None:
                self.capture.release()

            self.capture = test_cap
            self.current_source = new_source
            self.is_running = True

        self._set_status(f"Camera switched to {new_source}", ok=True)

    def restart_camera(self):
        self.start_camera()

    def _read_frame(self):
        with self.capture_lock:
            if self.capture is None or not self.capture.isOpened():
                return False, None
            return self.capture.read()

    def _get_embeddings_cached(self):
        path = config.EMBEDDINGS_PATH
        if not os.path.exists(path):
            self.emb_cache = {}
            self.emb_mtime = -1.0
            return self.emb_cache

        mtime = os.path.getmtime(path)
        if mtime != self.emb_mtime:
            self.emb_cache = utils.load_embeddings()
            self.emb_mtime = mtime
        return self.emb_cache

    def _set_status(self, text: str, ok: bool):
        self.status_var.set(text)
        self.status_label.configure(fg="#0b8f3e" if ok else "#b42318")

    def _safe_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return frame[y1:y2, x1:x2]

    def _recognize(self, emb: np.ndarray) -> Tuple[str, float, Optional[int]]:
        known = self._get_embeddings_cached()
        user_id, score = utils.find_best_match(emb, known)
        if user_id is None:
            return "Unknown", 0.0, None
        user = database.get_user_by_id(user_id)
        if not user:
            return "Unknown", 0.0, None
        return user["name"], score, user_id

    def _eye_aspect_ratio(self, eye: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = eye
        a = np.linalg.norm(p2 - p6)
        b = np.linalg.norm(p3 - p5)
        c = np.linalg.norm(p1 - p4) + 1e-6
        return float((a + b) / (2.0 * c))

    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _eye_aspect_ratio_mediapipe(self, landmarks, indices, width: int, height: int) -> float:
        points = np.array(
            [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
            dtype=np.float32,
        )
        v1 = self._euclidean_distance(points[1], points[5])
        v2 = self._euclidean_distance(points[2], points[4])
        h = self._euclidean_distance(points[0], points[3])
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _get_blink_ear_from_mediapipe(self, frame: np.ndarray) -> Optional[float]:
        if not self.use_mediapipe_blink:
            return None

        if self.mediapipe_mode == "solutions" and self.mp_face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            lmk = results.multi_face_landmarks[0].landmark
        elif self.mediapipe_mode == "tasks" and self.mp_face_landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp_image_cls(image_format=self.mp_image_format.SRGB, data=rgb)
            results = self.mp_face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            lmk = results.face_landmarks[0]
        else:
            return None

        h, w = frame.shape[:2]
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]
        left_ear = self._eye_aspect_ratio_mediapipe(lmk, left_eye_idx, w, h)
        right_ear = self._eye_aspect_ratio_mediapipe(lmk, right_eye_idx, w, h)
        return float((left_ear + right_ear) / 2.0)

    def _compute_ear_from_lmk68(self, lmk68: Optional[np.ndarray]) -> Optional[float]:
        if lmk68 is None or lmk68.shape[0] < 48:
            return None

        pts = lmk68[:, :2]
        left_eye = pts[36:42]
        right_eye = pts[42:48]
        return float((self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0)

    def _update_blink_count(self, user_id: int, ear: Optional[float]) -> int:
        now = time.time()
        state = self.blink_state.get(
            user_id,
            {
                "blink_count": 0,
                "eyes_closed": False,
                "closed_frames": 0,
                "first_blink_time": None,
                "last_seen": now,
            },
        )
        state["last_seen"] = now

        if ear is None:
            self.blink_state[user_id] = state
            return int(state["blink_count"])

        closed_threshold = config.EYE_AR_CLOSED_THRESHOLD
        open_threshold = config.EYE_AR_OPEN_THRESHOLD

        if ear <= closed_threshold:
            state["eyes_closed"] = True
            state["closed_frames"] = int(state.get("closed_frames", 0)) + 1
        elif ear >= open_threshold and state["eyes_closed"]:
            state["eyes_closed"] = False
            closed_frames = int(state.get("closed_frames", 0))
            state["closed_frames"] = 0
            if closed_frames >= config.BLINK_MIN_CLOSED_FRAMES:
                first_blink_time = state.get("first_blink_time")
                if first_blink_time is None or (now - first_blink_time > config.BLINK_SEQUENCE_TIMEOUT_SECONDS):
                    state["first_blink_time"] = now
                    state["blink_count"] = 1
                else:
                    state["blink_count"] += 1
        elif ear > closed_threshold:
            state["closed_frames"] = 0

        first_blink_time = state.get("first_blink_time")
        if first_blink_time is not None and now - first_blink_time > config.BLINK_SEQUENCE_TIMEOUT_SECONDS:
            state["blink_count"] = 0
            state["first_blink_time"] = None

        self.blink_state[user_id] = state
        return int(state["blink_count"])

    def _cleanup_blink_state(self):
        now = time.time()
        stale = []
        for user_id, state in self.blink_state.items():
            if now - float(state.get("last_seen", now)) > config.BLINK_TRACK_CLEANUP_SECONDS:
                stale.append(user_id)
        for user_id in stale:
            self.blink_state.pop(user_id, None)

    def _process_frame(self, frame):
        faces = embeddings.get_all_face_data(frame)
        self._cleanup_blink_state()
        frame_ear = self._get_blink_ear_from_mediapipe(frame)

        if not faces:
            cv2.putText(frame, "No Face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            return frame

        for face in faces:
            bbox = face["bbox"]
            emb = face["embedding"]
            lmk68 = face.get("landmark_3d_68")

            x1, y1, x2, y2 = bbox
            name, sim, user_id = self._recognize(emb)

            if user_id is not None and sim >= config.IDENTITY_CONFIDENCE_THRESHOLD:
                ear = frame_ear
                if ear is None:
                    ear = self._compute_ear_from_lmk68(lmk68)
                blink_count = self._update_blink_count(user_id, ear)

                if database.check_attendance_today(user_id):
                    label = f"{name} ({sim:.2f}) - Already Marked"
                elif ear is None:
                    label = f"{name} ({sim:.2f}) - Blink unavailable"
                elif blink_count >= config.BLINKS_REQUIRED_FOR_ATTENDANCE:
                    marked = database.mark_attendance(user_id)
                    label = f"{name} ({sim:.2f}) - {'Marked' if marked else 'Already Marked'}"
                    self.blink_state.pop(user_id, None)
                else:
                    label = (
                        f"{name} ({sim:.2f}) - Blink {config.BLINKS_REQUIRED_FOR_ATTENDANCE}x "
                        f"[{blink_count}/{config.BLINKS_REQUIRED_FOR_ATTENDANCE}]"
                    )

                color = (0, 190, 0)
                if ear is not None:
                    cv2.putText(
                        frame,
                        f"EAR:{ear:.2f}",
                        (x1, min(frame.shape[0] - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (30, 144, 255),
                        1,
                    )
            else:
                label = "Unknown/Spoof"
                color = (0, 90, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                color,
                2,
            )

        return frame

    def video_loop(self):
        ok, frame = self._read_frame()
        if not ok or frame is None:
            if self.is_running:
                self._set_status("Camera disconnected. Attempting reconnect...", ok=False)
                self.start_camera()
            frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            self.frame_count += 1
            if self.frame_count % max(1, config.TK_PROCESS_EVERY_N_FRAMES) == 0:
                frame = self._process_frame(frame)
                self.last_frame = frame.copy()
            elif self.last_frame is not None:
                frame = self.last_frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(config.TK_UI_REFRESH_MS, self.video_loop)

    def refresh_dashboard(self):
        users = database.get_all_users()
        today = database.get_today_attendance()

        total = len(users)
        present = len(today)
        absent = max(0, total - present)

        self.total_var.set(str(total))
        self.present_var.set(str(present))
        self.absent_var.set(str(absent))

        for item in self.table.get_children():
            self.table.delete(item)
        for row in today:
            self.table.insert("", "end", values=(row["name"], row["time"], row["status"]))

        self.root.after(config.TK_STATS_REFRESH_MS, self.refresh_dashboard)

    def _open_enroll_window(self, name: str):
        if self.enroll_window is not None and self.enroll_window.winfo_exists():
            self.enroll_window.destroy()

        self.enroll_progress_var.set(0.0)
        self.enroll_status_var.set("Starting capture...")
        self.enroll_tip_var.set("Keep your face centered, stable, and well lit.")

        win = tk.Toplevel(self.root)
        win.title("Enrollment Progress")
        win.geometry("480x230")
        win.resizable(False, False)
        win.configure(bg="#ffffff")
        win.transient(self.root)
        win.grab_set()

        frame = ttk.Frame(win, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=f"Enrolling: {name}", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=f"Collecting {config.MIN_ENROLLMENT_IMAGES} live samples",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        bar = ttk.Progressbar(
            frame,
            variable=self.enroll_progress_var,
            maximum=config.MIN_ENROLLMENT_IMAGES,
            mode="determinate",
            length=420,
        )
        bar.pack(anchor="w", pady=(4, 8))

        ttk.Label(frame, textvariable=self.enroll_status_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, textvariable=self.enroll_tip_var, style="SubTitle.TLabel", wraplength=430).pack(anchor="w", pady=(6, 0))

        self.enroll_window = win

    def _update_enroll_window(self, captured: int, required: int, attempts: int, confidence: float):
        self.enroll_progress_var.set(float(captured))
        self.enroll_status_var.set(f"Captured {captured}/{required} | Attempts: {attempts} | Live confidence: {confidence:.2f}")
        if confidence < config.ENROLLMENT_LIVENESS_THRESHOLD:
            self.enroll_tip_var.set("Liveness is low. Increase front light and avoid fast head movement.")
        else:
            self.enroll_tip_var.set("Great. Hold steady for next sample.")

    def _close_enroll_window(self):
        if self.enroll_window is not None and self.enroll_window.winfo_exists():
            self.enroll_window.grab_release()
            self.enroll_window.destroy()
        self.enroll_window = None

    def enroll_user(self):
        if self.is_enrolling:
            return

        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Validation", "Please enter full name.")
            return

        self.is_enrolling = True
        self.enroll_btn.config(state="disabled")
        self._set_status("Enrollment started. Keep face centered and steady...", ok=True)
        self._open_enroll_window(name)

        thread = threading.Thread(target=self._enroll_worker, args=(name,), daemon=True)
        thread.start()

    def _enroll_worker(self, name: str):
        required = config.MIN_ENROLLMENT_IMAGES
        samples = []
        emb_list = []
        attempts = 0
        max_attempts = 140

        while len(samples) < required and attempts < max_attempts:
            attempts += 1
            ok, frame = self._read_frame()
            if not ok or frame is None:
                time.sleep(0.12)
                continue

            faces = embeddings.get_all_face_data(frame)
            if not faces:
                time.sleep(0.12)
                continue

            face = max(
                faces,
                key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
            )
            bbox = face["bbox"]
            emb = face["embedding"]

            crop = self._safe_crop(frame, bbox)
            is_live, conf = liveness.check_liveness(crop, threshold=config.ENROLLMENT_LIVENESS_THRESHOLD)
            self.root.after(
                0,
                lambda c=len(samples), r=required, a=attempts, conf_val=conf: self._update_enroll_window(c, r, a, conf_val),
            )

            if not is_live:
                logger.warning("Enrollment capture rejected (%.2f)", conf)
                time.sleep(0.15)
                continue

            samples.append(frame.copy())
            emb_list.append(emb)
            self.root.after(
                0,
                lambda c=len(samples), r=required, a=attempts, conf_val=conf: self._update_enroll_window(c, r, a, conf_val),
            )
            self.root.after(0, lambda c=len(samples), r=required: self._set_status(f"Captured {c}/{r} samples...", ok=True))
            time.sleep(0.3)

        if len(emb_list) < required:
            self.root.after(0, lambda: self._enroll_done(False, "Could not capture enough valid live samples."))
            return

        user_id = database.add_user(name)
        avg_embedding = np.mean(np.stack(emb_list, axis=0), axis=0).astype(np.float32)

        emb_store = utils.load_embeddings()
        emb_store[user_id] = avg_embedding
        utils.save_embeddings(emb_store)

        user_raw_dir = os.path.join(config.RAW_DATA_PATH, str(user_id))
        os.makedirs(user_raw_dir, exist_ok=True)
        for idx, img in enumerate(samples, start=1):
            cv2.imwrite(os.path.join(user_raw_dir, f"{idx}.jpg"), img)

        self.root.after(0, lambda: self._enroll_done(True, f"User '{name}' enrolled successfully."))

    def _enroll_done(self, success: bool, msg: str):
        self.is_enrolling = False
        self.enroll_btn.config(state="normal")
        self._close_enroll_window()
        if success:
            self.name_var.set("")
            self._set_status(msg, ok=True)
            messagebox.showinfo("Enrollment", msg)
            self.refresh_dashboard()
        else:
            self._set_status(msg, ok=False)
            messagebox.showerror("Enrollment", msg)

    def open_attendance_viewer(self):
        viewer_path = os.path.join(config.BASE_DIR, "attendance_viewer.py")
        if not os.path.exists(viewer_path):
            messagebox.showerror("Attendance Viewer", "attendance_viewer.py was not found.")
            return
        subprocess.Popen([sys.executable, viewer_path], cwd=config.BASE_DIR)

    def on_close(self):
        if self.mp_face_mesh is not None:
            try:
                self.mp_face_mesh.close()
            except Exception:
                pass
        if self.mp_face_landmarker is not None:
            try:
                self.mp_face_landmarker.close()
            except Exception:
                pass
        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
        self.root.destroy()


def main():
    root = tk.Tk()
    DesktopAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
