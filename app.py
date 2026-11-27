
import csv
import io
import os
import secrets
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui
from flask import (
    Flask, Response, jsonify, redirect, render_template,
    request, session, send_file, url_for, stream_with_context
)
from werkzeug.security import check_password_hash, generate_password_hash


# =========================================================
# ML MODEL IMPORTS
# =========================================================

try:
    from ml.features import extract_features
    from ml.predict import load_model, predict_gesture
except ImportError:
    from backend.ml.features import extract_features
    from backend.ml.predict import load_model, predict_gesture


# =========================================================
# PATHS AND FLASK CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DB_PATH = ROOT / "gestureguard.db"

app = Flask(__name__, template_folder=str(FRONTEND))

app.secret_key = (
    os.environ.get("GESTUREGUARD_SECRET")
    or secrets.token_hex(32)
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

ADMIN_USERNAME = os.environ.get(
    "GESTUREGUARD_ADMIN_USER",
    "siteadmin"
)

ADMIN_PASSWORD = os.environ.get(
    "GESTUREGUARD_ADMIN_PASSWORD",
    "Set-A-Strong-Admin-Password-2026!"
)


# =========================================================
# LOAD TRAINED ML MODEL
# =========================================================

ML_MODEL_AVAILABLE = False

try:
    load_model()
    ML_MODEL_AVAILABLE = True
    print("ML gesture model loaded successfully.")
except Exception as exc:
    print("ML model unavailable. Using fallback detection.")
    print("Reason:", exc)


# =========================================================
# MEDIAPIPE AND MOUSE CONFIGURATION
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()


# =========================================================
# PERFORMANCE CONFIGURATION
# =========================================================

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

JPEG_QUALITY = 55

# Run the ML classifier once every two captured frames.
ML_CLASSIFY_EVERY_N_FRAMES = 1

# Draw landmarks once every two frames.
DRAW_LANDMARKS_EVERY_N_FRAMES = 1

# Cursor smoothing: higher values respond faster.
CURSOR_ALPHA = 0.65

# Number of recent predictions used for gesture smoothing.
GESTURE_HISTORY_SIZE = 3

ML_CONFIDENCE_THRESHOLD = 0.55

AUTH_HOLD_SECONDS = 0.65
CLICK_COOLDOWN = 0.8


# =========================================================
# GESTURE CONFIGURATION
# =========================================================

GESTURES = [
    {
        "key": "OPEN_PALM",
        "label": "Open palm",
        "hint": "Show all five fingers"
    },
    {
        "key": "FIST",
        "label": "Closed fist",
        "hint": "Curl your fingers into a fist"
    },
    {
        "key": "PEACE",
        "label": "Peace sign",
        "hint": "Raise index and middle fingers"
    },
    {
        "key": "THUMB",
        "label": "Thumbs up",
        "hint": "Raise your thumb"
    },
]


# =========================================================
# SHARED APPLICATION STATE
# =========================================================

state_lock = threading.Lock()

state = {
    "phase": "logged_out",
    "message": "Log in to begin.",
    "running": False,
    "camera_ok": False,
    "authenticated": False,
    "auth_sequence": [],
    "auth_step": 0,
    "last_gesture": "None",
    "gesture_confidence": None,
    "ml_model_available": ML_MODEL_AVAILABLE,
    "error": "",
    "mouse_paused": False,
    "scroll_delta": 0,
}

latest_jpeg = None

stop_event = threading.Event()
camera_thread = None


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def init_db():
    with db_connect() as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                event TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            )
        """)

        row = db.execute(
            "SELECT id FROM users WHERE username=?",
            ("admin",)
        ).fetchone()

        if row is None:
            now = utc_now()

            db.execute("""
                INSERT INTO users(
                    username,
                    password_hash,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?)
            """, (
                "admin",
                generate_password_hash("ChangeMe123!"),
                now,
                now
            ))


def log_event(username, event, detail=""):
    try:
        with db_connect() as db:
            db.execute("""
                INSERT INTO audit_log(
                    username,
                    event,
                    detail,
                    created_at
                )
                VALUES(?,?,?,?)
            """, (
                username,
                event,
                detail[:500],
                utc_now()
            ))
    except sqlite3.Error:
        pass


# =========================================================
# LOGIN AND SESSION HELPERS
# =========================================================

def current_user():
    return session.get("username")


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        if not current_user():

            if request.path.startswith("/api/"):
                return jsonify(
                    ok=False,
                    message="Please log in first."
                ), 401

            return redirect(url_for("index"))

        return fn(*args, **kwargs)

    return wrapper


def gesture_label(key):
    return next(
        (g["label"] for g in GESTURES if g["key"] == key),
        key or "None"
    )


def get_state():
    with state_lock:
        result = dict(state)

    result["expected_gesture"] = (
        result["auth_sequence"][result["auth_step"]]
        if result["auth_step"] < len(result["auth_sequence"])
        else None
    )

    result["auth_sequence_labels"] = [
        gesture_label(g)
        for g in result["auth_sequence"]
    ]

    result["expected_label"] = (
        gesture_label(result["expected_gesture"])
        if result["expected_gesture"]
        else None
    )

    return result


def update_state(**kwargs):
    with state_lock:
        state.update(kwargs)


# =========================================================
# FALLBACK GESTURE DETECTOR
# =========================================================

def detect_gesture(hand):

    lm = hand.landmark

    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    thumb = abs(lm[4].x - lm[2].x) > 0.065

    if index and middle and ring and pinky:
        return "OPEN_PALM"

    if not any((index, middle, ring, pinky)):
        return "FIST"

    if index and middle and not ring and not pinky:
        return "PEACE"

    if thumb and not any((index, middle, ring, pinky)):
        return "THUMB"

    pinch = (
        (lm[4].x - lm[8].x) ** 2
        + (lm[4].y - lm[8].y) ** 2
    ) ** 0.5

    if pinch < 0.045:
        return "PINCH"

    return "OTHER"


# =========================================================
# ML CLASSIFICATION AND GESTURE SMOOTHING
# =========================================================

def classify_hand(hand, prediction_history):

    global ML_MODEL_AVAILABLE

    confidence = None

    if ML_MODEL_AVAILABLE:

        try:
            features = extract_features(hand)

            predicted_gesture, confidence = predict_gesture(
                features
            )

            if confidence < ML_CONFIDENCE_THRESHOLD:
                predicted_gesture = "OTHER"

        except Exception as exc:

            print("ML prediction failed:", exc)

            ML_MODEL_AVAILABLE = False
            prediction_history.clear()

            predicted_gesture = detect_gesture(hand)
            confidence = None

    else:
        predicted_gesture = detect_gesture(hand)

    prediction_history.append(predicted_gesture)

    counts = Counter(prediction_history)
    highest_count = max(counts.values())

    smoothed_gesture = None

    for item in reversed(prediction_history):

        if counts[item] == highest_count:
            smoothed_gesture = item
            break

    return smoothed_gesture, confidence


# =========================================================
# WEBCAM PROCESSING THREAD
# =========================================================

def camera_worker(username):

    global latest_jpeg

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    )

    if not cap.isOpened():

        update_state(
            phase="error",
            running=False,
            camera_ok=False,
            error=(
                "Could not open webcam. Check camera permissions "
                "and close other apps using it."
            ),
            message="Webcam could not be opened."
        )

        log_event(
            username,
            "WEBCAM_ERROR",
            "Could not open camera"
        )

        return

    # Request lower-resolution webcam frames.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # Request a small camera buffer to reduce stale frames.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Request MJPEG when supported by the camera driver.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    tracker = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60
    )

    last_accepted = 0.0
    last_click = 0.0

    smooth_x = None
    smooth_y = None

    previous_scroll_y = None
    paused = False

    prediction_history = deque(
        maxlen=GESTURE_HISTORY_SIZE
    )

    # Cached classification avoids running the model on every frame.
    frame_counter = 0
    cached_gesture = "OTHER"
    cached_confidence = None

    update_state(
        camera_ok=True,
        running=True,
        ml_model_available=ML_MODEL_AVAILABLE
    )

    log_event(
        username,
        "CAMERA_STARTED",
        "Optimized webcam stream started"
    )

    try:

        while not stop_event.is_set():

            # ---------------------------------------------
            # CAPTURE FRAME
            # ---------------------------------------------

            ok, frame = cap.read()

            if not ok:
                time.sleep(0.02)
                continue

            frame_counter += 1

            frame = cv2.flip(frame, 1)

            # ---------------------------------------------
            # MEDIAPIPE PROCESSING
            # ---------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            result = tracker.process(rgb)

            rgb.flags.writeable = True

            s = get_state()

            gesture = "None"
            confidence = None

            # ---------------------------------------------
            # HAND DETECTED
            # ---------------------------------------------

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                # Draw landmarks only on alternate frames.
                if (
                    frame_counter
                    % DRAW_LANDMARKS_EVERY_N_FRAMES == 0
                ):

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                # -----------------------------------------
                # ML CLASSIFICATION THROTTLING
                # -----------------------------------------

                should_classify = (
                    frame_counter
                    % ML_CLASSIFY_EVERY_N_FRAMES == 0
                    or not prediction_history
                )

                if should_classify:

                    cached_gesture, cached_confidence = (
                        classify_hand(
                            hand,
                            prediction_history
                        )
                    )

                gesture = cached_gesture
                confidence = cached_confidence

                update_state(
                    last_gesture=gesture,
                    gesture_confidence=(
                        round(confidence, 3)
                        if confidence is not None
                        else None
                    ),
                    ml_model_available=ML_MODEL_AVAILABLE
                )

                now = time.time()

                # -----------------------------------------
                # RANDOM HAND AUTHENTICATION
                # -----------------------------------------

                if s["phase"] == "hand_auth":

                    expected = s["expected_gesture"]

                    if (
                        gesture == expected
                        and now - last_accepted >= AUTH_HOLD_SECONDS
                    ):

                        last_accepted = now
                        new_step = s["auth_step"] + 1

                        if new_step >= len(s["auth_sequence"]):

                            update_state(
                                authenticated=True,
                                phase="mouse",
                                auth_step=new_step,
                                message=(
                                    "Hand verification successful. "
                                    "Use your hand to control the cursor."
                                )
                            )

                            log_event(
                                username,
                                "HAND_AUTH_SUCCESS",
                                "Random gesture challenge passed"
                            )

                        else:

                            update_state(
                                auth_step=new_step,
                                message=(
                                    "Correct. Next gesture: "
                                    + gesture_label(
                                        s["auth_sequence"][new_step]
                                    )
                                    + "."
                                )
                            )

                    elif gesture != expected:

                        update_state(
                            message=(
                                "Please show: "
                                + gesture_label(expected)
                            )
                        )

                # -----------------------------------------
                # MOUSE CONTROL
                # -----------------------------------------

                elif (
                    s["phase"] == "mouse"
                    and s["authenticated"]
                ):

                    if gesture == "FIST":

                        paused = True
                        previous_scroll_y = None

                    elif gesture == "OPEN_PALM":

                        paused = False
                        previous_scroll_y = None

                    update_state(mouse_paused=paused)

                    if not paused:

                        # ---------------------------------
                        # PEACE SIGN SCROLLING
                        # ---------------------------------

                        if gesture == "PEACE":

                            scroll_y = hand.landmark[9].y
                            scroll_delta_px = 0

                            if previous_scroll_y is not None:

                                delta = (
                                    previous_scroll_y - scroll_y
                                )

                                if abs(delta) >= 0.004:

                                    scroll_delta_px = int(
                                        max(
                                            -34,
                                            min(34, delta * 1500)
                                        )
                                    )

                                    if scroll_delta_px == 0:

                                        scroll_delta_px = (
                                            1 if delta > 0 else -1
                                        )

                                    try:

                                        scroll_amount = int(
                                            delta * 500
                                        )

                                        scroll_amount = max(
                                            -12,
                                            min(12, scroll_amount)
                                        )

                                        if scroll_amount != 0:
                                            pyautogui.scroll(
                                                scroll_amount
                                            )

                                    except Exception:
                                        pass

                            update_state(
                                scroll_delta=scroll_delta_px
                            )

                            previous_scroll_y = scroll_y

                        # ---------------------------------
                        # CURSOR MOVEMENT AND PINCH CLICK
                        # ---------------------------------

                        else:

                            previous_scroll_y = None

                            update_state(scroll_delta=0)

                            tip = hand.landmark[8]

                            tx = int(
                                max(
                                    0,
                                    min(
                                        1,
                                        (tip.x - 0.08) / 0.84
                                    )
                                ) * SCREEN_W
                            )

                            ty = int(
                                max(
                                    0,
                                    min(
                                        1,
                                        (tip.y - 0.08) / 0.84
                                    )
                                ) * SCREEN_H
                            )

                            # Faster cursor response.
                            smooth_x = (
                                tx
                                if smooth_x is None
                                else int(
                                    CURSOR_ALPHA * tx
                                    + (1 - CURSOR_ALPHA) * smooth_x
                                )
                            )

                            smooth_y = (
                                ty
                                if smooth_y is None
                                else int(
                                    CURSOR_ALPHA * ty
                                    + (1 - CURSOR_ALPHA) * smooth_y
                                )
                            )

                            try:

                                pyautogui.moveTo(
                                    smooth_x,
                                    smooth_y,
                                    duration=0
                                )

                            except pyautogui.FailSafeException:

                                paused = True

                                update_state(
                                    mouse_paused=True,
                                    message=(
                                        "Safety pause triggered. "
                                        "Move hand away from screen corner."
                                    )
                                )

                            # ---------------------------------
                            # PINCH-TO-CLICK
                            # ---------------------------------

                            if (
                                gesture == "PINCH"
                                and now - last_click > CLICK_COOLDOWN
                            ):

                                try:

                                    pyautogui.click()
                                    last_click = now

                                except Exception as exc:
                                    print("Mouse click error:", exc)

            # ---------------------------------------------
            # NO HAND DETECTED
            # ---------------------------------------------

            else:

                prediction_history.clear()

                cached_gesture = "OTHER"
                cached_confidence = None

                update_state(
                    last_gesture="None",
                    gesture_confidence=None,
                    scroll_delta=0
                )

                smooth_x = None
                smooth_y = None
                previous_scroll_y = None

            # ---------------------------------------------
            # DRAW CAMERA STATUS
            # ---------------------------------------------

            s = get_state()

            overlay_width = frame.shape[1] - 16

            cv2.rectangle(
                frame,
                (8, 8),
                (overlay_width, 92),
                (250, 252, 255),
                -1
            )

            cv2.putText(
                frame,
                "GestureGuard | " + s["phase"].upper(),
                (20, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (45, 70, 95),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                s["message"][:55],
                (20, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (45, 70, 95),
                1,
                cv2.LINE_AA
            )

            if confidence is not None:

                confidence_text = (
                    f"ML confidence: {confidence:.0%}"
                )

            else:

                confidence_text = (
                    "Fallback gesture detection"
                    if not ML_MODEL_AVAILABLE
                    else "ML gesture prediction"
                )

            cv2.putText(
                frame,
                confidence_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (45, 70, 95),
                1,
                cv2.LINE_AA
            )

            # ---------------------------------------------
            # JPEG ENCODING
            # ---------------------------------------------

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    JPEG_QUALITY
                ]
            )

            if ok:

                with state_lock:
                    latest_jpeg = jpg.tobytes()

    except Exception as exc:

        update_state(
            phase="error",
            running=False,
            error=str(exc),
            message="Camera processing failed."
        )

        log_event(
            username,
            "RUNTIME_ERROR",
            str(exc)
        )

        print("Camera processing error:", exc)

    finally:

        cap.release()
        tracker.close()

        with state_lock:

            state["camera_ok"] = False
            state["running"] = False

            latest_jpeg = None


# =========================================================
# STOP CAMERA
# =========================================================

def stop_camera():

    stop_event.set()

    update_state(
        running=False,
        camera_ok=False,
        authenticated=False,
        phase="stopped",
        mouse_paused=True,
        message=(
            "Camera stopped. Start again to get a new "
            "random hand challenge."
        )
    )

    try:
        pyautogui.mouseUp()
    except Exception:
        pass


# =========================================================
# MAIN PAGE
# =========================================================

@app.get("/")
def index():

    return render_template(
        "index.html",
        username=current_user()
    )


# =========================================================
# STATUS API
# =========================================================

@app.get("/api/status")
@login_required
def api_status():

    s = get_state()

    s["username"] = current_user()
    s["ml_model_available"] = ML_MODEL_AVAILABLE

    return jsonify(s)


# =========================================================
# USER LOGIN
# =========================================================

@app.post("/api/login")
def api_login():

    data = request.get_json(silent=True) or {}

    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    with db_connect() as db:

        user = db.execute("""
            SELECT username, password_hash
            FROM users
            WHERE username=?
        """, (username,)).fetchone()

    if (
        not user
        or not check_password_hash(
            user["password_hash"],
            password
        )
    ):

        log_event(
            username or None,
            "LOGIN_FAILED",
            "Invalid username or password"
        )

        return jsonify(
            ok=False,
            message="Incorrect username or password."
        ), 401

    session.clear()
    session["username"] = user["username"]

    sequence = [
        g["key"]
        for g in secrets.SystemRandom().sample(
            GESTURES[:3],
            k=3
        )
    ]

    update_state(
        phase="logged_in",
        authenticated=False,
        auth_sequence=sequence,
        auth_step=0,
        message=(
            "Login successful. Start the webcam and "
            "follow your random hand challenge."
        ),
        last_gesture="None",
        gesture_confidence=None,
        error="",
        mouse_paused=False
    )

    log_event(
        username,
        "LOGIN_SUCCESS",
        "Website login succeeded; new random hand challenge created"
    )

    return jsonify(
        ok=True,
        message=(
            "Login successful. A new random hand challenge is ready."
        ),
        challenge=[
            {
                "label": gesture_label(g),
                "hint": next(
                    x["hint"]
                    for x in GESTURES
                    if x["key"] == g
                )
            }
            for g in sequence
        ]
    )


# =========================================================
# START CAMERA API
# =========================================================

@app.post("/api/start")
@login_required
def api_start():

    if get_state()["running"]:

        return jsonify(
            ok=True,
            message="Webcam is already running."
        )

    s = get_state()

    if not s["auth_sequence"]:

        sequence = [
            g["key"]
            for g in secrets.SystemRandom().sample(
                GESTURES[:3],
                k=3
            )
        ]

        update_state(
            auth_sequence=sequence,
            auth_step=0
        )

    stop_event.clear()

    update_state(
        phase="hand_auth",
        authenticated=False,
        auth_step=0,
        error="",
        message=(
            f"Show {gesture_label(get_state()['auth_sequence'][0])} "
            "to begin."
        ),
        running=True
    )

    global camera_thread

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(current_user(),),
        daemon=True
    )

    camera_thread.start()

    return jsonify(
        ok=True,
        message="Webcam started inside the dashboard."
    )


# =========================================================
# STOP CAMERA API
# =========================================================

@app.post("/api/stop")
@login_required
def api_stop():

    log_event(
        current_user(),
        "CAMERA_STOPPED",
        "User stopped camera"
    )

    stop_camera()

    return jsonify(
        ok=True,
        message="Camera stopped."
    )


# =========================================================
# LOGOUT API
# =========================================================

@app.post("/api/logout")
@login_required
def api_logout():

    username = current_user()

    stop_camera()

    log_event(
        username,
        "LOGOUT",
        "User logged out"
    )

    session.clear()

    update_state(
        phase="logged_out",
        auth_sequence=[],
        auth_step=0,
        message="Log in to begin."
    )

    return jsonify(
        ok=True,
        message="Logged out."
    )


# =========================================================
# CHANGE PASSWORD API
# =========================================================

@app.post("/api/change-password")
@login_required
def change_password():

    data = request.get_json(silent=True) or {}

    old = str(data.get("current_password", ""))
    new = str(data.get("new_password", ""))

    if len(new) < 10:

        return jsonify(
            ok=False,
            message=(
                "New password must contain at least "
                "10 characters."
            )
        ), 400

    username = current_user()

    with db_connect() as db:

        row = db.execute("""
            SELECT password_hash
            FROM users
            WHERE username=?
        """, (username,)).fetchone()

        if (
            not row
            or not check_password_hash(
                row["password_hash"],
                old
            )
        ):

            return jsonify(
                ok=False,
                message="Current password is incorrect."
            ), 400

        db.execute("""
            UPDATE users
            SET password_hash=?, updated_at=?
            WHERE username=?
        """, (
            generate_password_hash(new),
            utc_now(),
            username
        ))

    log_event(
        username,
        "PASSWORD_CHANGED",
        "User changed password"
    )

    return jsonify(
        ok=True,
        message="Password changed successfully."
    )


# =========================================================
# MOUSE CONTROL PAGE
# =========================================================

@app.get("/mouse")
@login_required
def mouse_page():

    s = get_state()

    if s["phase"] != "mouse" or not s["authenticated"]:
        return redirect(url_for("index"))

    return render_template(
        "mouse.html",
        username=current_user()
    )


# =========================================================
# ADMIN PAGE
# =========================================================

@app.get("/admin")
def admin_page():

    return render_template("admin.html")


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.post("/api/admin/login")
def admin_login():

    data = request.get_json(silent=True) or {}

    username = str(data.get("username", ""))
    password = str(data.get("password", ""))

    if (
        not secrets.compare_digest(username, ADMIN_USERNAME)
        or not secrets.compare_digest(password, ADMIN_PASSWORD)
    ):

        log_event(
            "ADMIN",
            "ADMIN_LOGIN_FAILED",
            "Invalid admin credentials"
        )

        return jsonify(
            ok=False,
            message="Invalid admin username or password."
        ), 401

    session["admin_authenticated"] = True

    log_event(
        "ADMIN",
        "ADMIN_LOGIN_SUCCESS",
        "Administrator signed in"
    )

    return jsonify(
        ok=True,
        message="Admin sign-in successful."
    )


# =========================================================
# ADMIN AUTHORIZATION
# =========================================================

def admin_required(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):

        if not session.get("admin_authenticated"):

            return jsonify(
                ok=False,
                message="Admin sign-in required."
            ), 401

        return fn(*args, **kwargs)

    return wrapper


# =========================================================
# ADMIN LOGOUT
# =========================================================

@app.post("/api/admin/logout")
def admin_logout():

    if session.get("admin_authenticated"):

        log_event(
            "ADMIN",
            "ADMIN_LOGOUT",
            "Administrator signed out"
        )

    session.pop("admin_authenticated", None)

    return jsonify(ok=True)


# =========================================================
# ADMIN SUMMARY
# =========================================================

@app.get("/api/admin/summary")
@admin_required
def admin_summary():

    with db_connect() as db:

        users = db.execute("""
            SELECT username, created_at, updated_at
            FROM users
            ORDER BY created_at DESC
        """).fetchall()

        events = db.execute("""
            SELECT username, event, detail, created_at
            FROM audit_log
            ORDER BY id DESC
            LIMIT 500
        """).fetchall()

        counts = db.execute("""
            SELECT event, COUNT(*) AS total
            FROM audit_log
            GROUP BY event
            ORDER BY total DESC
        """).fetchall()

    return jsonify(
        ok=True,
        users=[dict(r) for r in users],
        events=[dict(r) for r in events],
        counts=[dict(r) for r in counts]
    )


# =========================================================
# ADMIN CSV REPORT
# =========================================================

@app.get("/api/admin/report.csv")
@admin_required
def admin_report_csv():

    with db_connect() as db:

        rows = db.execute("""
            SELECT a.created_at, a.username, a.event, a.detail
            FROM audit_log a
            ORDER BY a.id DESC
        """).fetchall()

        users = db.execute("""
            SELECT username, created_at, updated_at
            FROM users
            ORDER BY username
        """).fetchall()

    out = io.StringIO()
    writer = csv.writer(out)

    writer.writerow([
        "Record type",
        "Timestamp (UTC)",
        "Username",
        "Event",
        "Details",
        "Account created",
        "Account updated"
    ])

    for user in users:

        writer.writerow([
            "USER_ACCOUNT",
            "",
            user["username"],
            "REGISTERED_USER",
            "Registered user account",
            user["created_at"],
            user["updated_at"]
        ])

    for row in rows:

        writer.writerow([
            "ACTIVITY",
            row["created_at"],
            row["username"],
            row["event"],
            row["detail"],
            "",
            ""
        ])

    payload = io.BytesIO(
        out.getvalue().encode("utf-8-sig")
    )

    payload.seek(0)

    return send_file(
        payload,
        mimetype="text/csv",
        as_attachment=True,
        download_name=(
            "gestureguard_aggregate_activity_report.csv"
        )
    )


# =========================================================
# WEBCAM VIDEO STREAM
# =========================================================

@app.get("/video_feed")
@login_required
def video_feed():

    def generate():

        while not stop_event.is_set():

            with state_lock:
                frame = latest_jpeg

            if frame is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            else:
                time.sleep(0.05)

    return Response(
        stream_with_context(generate()),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        ),
        headers={
            "Cache-Control": (
                "no-cache, no-store, must-revalidate"
            ),
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_db()


# =========================================================
# START FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    print("GestureGuard: http://127.0.0.1:5000")

    print(
        "Demo user login: admin / ChangeMe123! "
        "(change it after signing in)."
    )

    print(
        "Separate admin page: "
        "http://127.0.0.1:5000/admin"
    )

    print("Admin username: " + ADMIN_USERNAME)

    print(
        "Set GESTUREGUARD_ADMIN_PASSWORD in your "
        "terminal environment to change the admin password."
    )

    print("ML model available:", ML_MODEL_AVAILABLE)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
