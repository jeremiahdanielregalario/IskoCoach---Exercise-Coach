import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from av import VideoFrame
import queue
import pandas as pd
import altair as alt
import threading
import json
import os
import time 

# ----------------------
# Utility: file load/save
# ----------------------
def load_users():
    """Load users from a JSON file"""
    try:
        if os.path.exists("users.json"):
            with open("users.json", "r") as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_users(users):
    """Save users to a JSON file"""
    try:
        with open("users.json", "w") as f:
            json.dump(users, f)
    except:
        pass

def load_workout_data():
    """Load workout data from a JSON file"""
    try:
        if os.path.exists("workout_data.json"):
            with open("workout_data.json", "r") as f:
                return json.load(f)
        return []
    except:
        return []

def save_workout_data(data):
    """Save workout data to a JSON file"""
    try:
        with open("workout_data.json", "w") as f:
            json.dump(data, f, default=str)  # default=str handles timestamps
    except:
        pass

# ----------------------
# Page config & session defaults
# ----------------------
st.set_page_config(page_title="IskoCoach Prototype", layout="centered")

st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("current_user", None)
st.session_state.setdefault("start_squats", False)
st.session_state.setdefault("target_reps", 10)
st.session_state.setdefault("workout_result_queue", queue.Queue())

if "users" not in st.session_state:
    st.session_state.users = load_users()

if "workout_data" not in st.session_state:
    st.session_state.workout_data = load_workout_data()

# ----------------------
# Login/Register
# ----------------------
def login_screen():
    st.title("🏋️ IskoCoach Login")
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success("Logged in!")
            st.rerun()
        else:
            st.error("Invalid username/password")

    st.subheader("Register")
    new_user = st.text_input("New Username", key="reg_username")
    new_pass = st.text_input("New Password", type="password", key="reg_password")
    if st.button("Register"):
        if not new_user:
            st.error("Please enter a username")
        elif new_user in st.session_state.users:
            st.error("Username already exists")
        else:
            st.session_state.users[new_user] = new_pass
            save_users(st.session_state.users)
            st.success("Registered! You can login now")

# ----------------------
# Squats Coach
# ----------------------
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def is_right_angle_at_knee(head, knee, opposite_knee):
    a = np.array(head)
    b = np.array(knee)
    c = np.array(opposite_knee)
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    return abs(dot) < 3000  # tolerance

class PoseCoach(VideoTransformerBase):
    def __init__(self, target_reps=10, result_queue=None):
        self.pose = mp_pose.Pose()
        self.knee_suggestion = ""
        self.back_suggestion = ""
        self.knee_color = (0, 255, 0)
        self.back_color = (0, 255, 0)
        self.reps = 0
        self.squat_state = "up"
        self.score = 100
        self.target_reps = target_reps
        self.finished = False
        self.in_position = False
        self.MIN_RATIO = 0.25
        self.MAX_RATIO = 0.7
        self.result_queue = result_queue
        self._workout_saved = False
        self._lock = threading.Lock()

        # Setup phase
        self.setup_mode = True  # Start in setup mode
        self.setup_start_time = None
        self.setup_countdown = 5  # 5 seconds to get in position
        self.setup_complete = False

        # Enhanced scoring system
        self.score = 0  # Start from 0
        self.points_per_rep = 100.0 / self.target_reps  # Calculate points per rep
        self.error_count = 0  # Track total errors
        self.errors_log = []  # Store error details
        self.current_rep_scored = False
        self.last_rep_count = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        small_img = cv2.resize(img, (320, 240))
        rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_small)

        self.knee_color = (0, 255, 0)
        self.back_color = (0, 255, 0)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            scale_x = w / 320
            scale_y = h / 240

            def to_original(p):
                return (int(p.x * 320 * scale_x), int(p.y * 240 * scale_y))

            head = to_original(lm[mp_pose.PoseLandmark.NOSE])
            left_hip = to_original(lm[mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = to_original(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            left_knee = to_original(lm[mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = to_original(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
            left_ankle = to_original(lm[mp_pose.PoseLandmark.LEFT_ANKLE])

            groin = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
            hip_to_ankle = abs(groin[1] - left_ankle[1])
            ratio = hip_to_ankle / h
            self.in_position = self.MIN_RATIO <= ratio <= self.MAX_RATIO

            knee_angle = calculate_angle(groin, left_knee, left_ankle)

            # SETUP PHASE (must stay in position for 5 seconds)
            if self.setup_mode and not self.setup_complete:
                if self.in_position:
                    if self.setup_start_time is None:
                        self.setup_start_time = time.time()
                    
                    elapsed = time.time() - self.setup_start_time
                    remaining = max(0, self.setup_countdown - elapsed)
                    
                    if remaining > 0:
                        # Show countdown
                        cv2.putText(img, f"Get in position... {remaining:.1f}s", (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        # Setup complete
                        self.setup_complete = True
                        self.setup_mode = False
                else:
                    # Reset if user moves out of position
                    self.setup_start_time = None
                    cv2.putText(img, "Please get into squat position", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Only start counting reps after setup is complete
            elif self.setup_complete and not self.finished:
                # Squat state and reps
                DOWN_THRESHOLD = 130
                UP_THRESHOLD = 160

                if self.in_position:
                    if self.squat_state == "up" and knee_angle < DOWN_THRESHOLD:
                        self.squat_state = "down"
                        self.current_rep_scored = False  # Reset for new rep
                    elif self.squat_state == "down" and knee_angle > UP_THRESHOLD:
                        self.squat_state = "up"
                        self.reps += 1
                        self.last_rep_count = self.reps
                        
                        # Add points for completing a rep
                        if not self.current_rep_scored:
                            self.score += self.points_per_rep
                            self.current_rep_scored = True

                self.knee_suggestion = ""
                self.back_suggestion = ""

                # Check for form mistakes and deduct points
                if self.squat_state == "down" and not self.current_rep_scored:
                    # Check if squat is too low (below 90 degrees)
                    if knee_angle < 90:
                        self.knee_suggestion = "⬆️ You're going too low!"
                        self.knee_color = (0, 0, 255)
                        # NEW: Deduct half the points per rep for this mistake
                        deduction = self.points_per_rep / 2
                        self.score = max(0, self.score - deduction)
                        self.error_count += 1
                        self.errors_log.append(f"Rep {self.reps+1}: Squat too low ({int(knee_angle)}°)")
                        self.current_rep_scored = True  # Mark as scored to avoid multiple deductions
                    else:
                        self.knee_suggestion = "✅ Good knee angle!"
                        self.knee_color = (0, 255, 0)

                # Back suggestion (only if knees good)
                if 90 <= knee_angle <= 130 and not self.current_rep_scored:
                    right_angle_left = is_right_angle_at_knee(head, left_knee, right_knee)
                    right_angle_right = is_right_angle_at_knee(head, right_knee, left_knee)
                    if right_angle_left or right_angle_right:
                        self.back_suggestion = "🧍 Keep your back straight!"
                        self.back_color = (0, 0, 255)
                        # NEW: Deduct half the points per rep for this mistake
                        deduction = self.points_per_rep / 2
                        self.score = max(0, self.score - deduction)
                        self.error_count += 1
                        self.errors_log.append(f"Rep {self.reps+1}: Bad back alignment")
                        self.current_rep_scored = True  # Mark as scored to avoid multiple deductions
                    else:
                        self.back_suggestion = "✅ Back alignment good!"
                        self.back_color = (0, 255, 0)
                else:
                    self.back_suggestion = ""

                # Check if workout is finished
                if self.reps >= self.target_reps and not self.finished:
                    with self._lock:
                        if not self._workout_saved:
                            self.finished = True
                            # Ensure final score is capped at 100
                            self.score = min(100, max(0, self.score))
                            # Put result in queue for main thread to process
                            if self.result_queue:
                                try:
                                    self.result_queue.put({
                                        "reps": self.reps,
                                        "score": round(self.score, 1),
                                        "errors": self.error_count,  # NEW: Add error count
                                        "errors_log": self.errors_log,  # NEW: Add error details
                                        "finished": True
                                    }, block=False)
                                except queue.Full:
                                    pass
                            self._workout_saved = True

            # Enhanced HUD display with error count
            cv2.putText(img, f"Reps: {self.reps}/{self.target_reps}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, f"Score: {round(self.score, 1)}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Errors: {self.error_count}", (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if self.setup_mode and not self.setup_complete:
                # Already showing setup message above
                pass
            elif self.finished:
                cv2.putText(img, "🏁 Finished!", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                if self.setup_complete:
                    cv2.putText(img, "✅ Ready! Start squatting", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if self.knee_suggestion:
                    cv2.putText(img, self.knee_suggestion, (20, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.knee_color, 2)
                if self.back_suggestion:
                    cv2.putText(img, self.back_suggestion, (20, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.back_color, 2)

            knee_circle_color = (0, 255, 0) if 90 <= knee_angle <= 130 else (0, 0, 255)
            cv2.circle(img, left_knee, 8, knee_circle_color, -1)
            cv2.putText(img, f"{int(knee_angle)}°",
                       (left_knee[0] + 10, left_knee[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, knee_circle_color, 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------
# Bicep Curl Coach
# ----------------------
class BicepCurlCoach(VideoTransformerBase):
    """
    Bicep curl coach - TRACKS BOTH ARMS
    """
    def __init__(self, target_reps=10, result_queue=None):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State & scoring - Now tracking both arms separately
        self.left_reps = 0
        self.right_reps = 0
        self.left_state = "down"
        self.right_state = "down"
        self.target_reps = int(target_reps or 10)
        self.result_queue = result_queue
        self._workout_saved = False
        self._lock = threading.Lock()

        # Setup phase
        self.setup_mode = True
        self.setup_start_time = None
        self.setup_countdown = 3
        self.setup_complete = False

        # Scoring system
        self.score = 0.0
        self.points_per_rep = (100.0 / self.target_reps) if self.target_reps > 0 else 0.0
        self.error_count = 0
        self.errors_log = []
        self.left_rep_scored = False
        self.right_rep_scored = False

        # Form feedback for both arms
        self.posture_suggestion = ""
        self.left_arm_suggestion = ""
        self.right_arm_suggestion = ""
        self.posture_color = (0, 255, 0)
        self.left_arm_color = (0, 255, 0)
        self.right_arm_color = (0, 255, 0)
        self.finished = False

        # Thresholds
        self.UP_THRESHOLD = 60
        self.DOWN_THRESHOLD = 160

        # Performance optimizations
        self._process_every_n_frames = 3
        self._frame_counter = 0
        self._last_process_time = 0.0
        self._inference_w = 256
        self._inference_h = 144
        
        # Cache for both arms
        self._cache = {
            "left_elbow_angle": 0.0,
            "right_elbow_angle": 0.0,
            "posture_angle": 180.0,
            "l_shoulder": None, "l_elbow": None, "l_wrist": None,
            "r_shoulder": None, "r_elbow": None, "r_wrist": None,
            "posture_suggestion": "",
            "left_arm_suggestion": "", "right_arm_suggestion": "",
            "posture_color": (0, 255, 0),
            "left_arm_color": (0, 255, 0), "right_arm_color": (0, 255, 0),
        }

    def _get_landmark_xy(self, lm_list, landmark_enum, scale_x, scale_y):
        """Safe landmark access"""
        try:
            idx = landmark_enum.value
            if idx >= len(lm_list):
                return None
            p = lm_list[idx]
            
            if hasattr(p, 'visibility') and p.visibility is not None:
                if p.visibility < 0.3:
                    return None
            
            if p.x is None or p.y is None:
                return None
                
            x = int(p.x * self._inference_w * scale_x)
            y = int(p.y * self._inference_h * scale_y)
            return (x, y)
        except Exception:
            return None

    def _check_verticality(self, shoulder, elbow):
        """Check if shoulder-elbow line is within 15° of vertical"""
        try:
            if shoulder is None or elbow is None:
                return 0, True  # Default to vertical
            
            dx = elbow[0] - shoulder[0]  # Horizontal
            dy = elbow[1] - shoulder[1]  # Vertical
            
            if abs(dy) < 1:
                return 90, False  # Horizontal line
            
            angle_rad = np.arctan(abs(dx) / abs(dy))
            angle_deg = np.degrees(angle_rad)
            
            # Check if within 15° tolerance
            is_vertical = angle_deg <= 15
            return angle_deg, is_vertical
        except Exception:
            return 0, True  # Default to vertical

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        self._frame_counter += 1
        now = time.time()
        should_process = (
            (self._frame_counter % self._process_every_n_frames == 0) and
            (now - self._last_process_time >= 0.05)
        )

        if should_process:
            try:
                small_img = cv2.resize(img, (self._inference_w, self._inference_h))
                rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_small)
                self._last_process_time = now

                if results and results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    scale_x = w / self._inference_w
                    scale_y = h / self._inference_h

                    # Get BOTH arms landmarks
                    l_shoulder = self._get_landmark_xy(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, scale_x, scale_y)
                    l_elbow = self._get_landmark_xy(lm, mp_pose.PoseLandmark.LEFT_ELBOW, scale_x, scale_y)
                    l_wrist = self._get_landmark_xy(lm, mp_pose.PoseLandmark.LEFT_WRIST, scale_x, scale_y)
                    
                    r_shoulder = self._get_landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, scale_x, scale_y)
                    r_elbow = self._get_landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_ELBOW, scale_x, scale_y)
                    r_wrist = self._get_landmark_xy(lm, mp_pose.PoseLandmark.RIGHT_WRIST, scale_x, scale_y)
                    
                    # Get posture landmarks (using left side for posture check)
                    l_hip = self._get_landmark_xy(lm, mp_pose.PoseLandmark.LEFT_HIP, scale_x, scale_y)
                    l_ankle = self._get_landmark_xy(lm, mp_pose.PoseLandmark.LEFT_ANKLE, scale_x, scale_y)

                    # Calculate angles for BOTH arms
                    left_elbow_angle = 0
                    right_elbow_angle = 0
                    posture_angle = 180
                    
                    if all([l_shoulder, l_elbow, l_wrist]):
                        left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    
                    if all([r_shoulder, r_elbow, r_wrist]):
                        right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    
                    if all([l_shoulder, l_hip, l_ankle]):
                        posture_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

                    # Update cache
                    self._cache.update({
                        "left_elbow_angle": left_elbow_angle,
                        "right_elbow_angle": right_elbow_angle,
                        "posture_angle": posture_angle,
                        "l_shoulder": l_shoulder, "l_elbow": l_elbow, "l_wrist": l_wrist,
                        "r_shoulder": r_shoulder, "r_elbow": r_elbow, "r_wrist": r_wrist,
                    })

                    # Setup phase (check if both arms are extended)
                    if self.setup_mode and not self.setup_complete:
                        # Both arms should be extended (>140°)
                        if left_elbow_angle > 140 and right_elbow_angle > 140:
                            if self.setup_start_time is None:
                                self.setup_start_time = time.time()
                            elapsed = time.time() - self.setup_start_time
                            remaining = max(0, self.setup_countdown - elapsed)
                            if remaining <= 0:
                                self.setup_complete = True
                                self.setup_mode = False
                        else:
                            self.setup_start_time = None

                    # Main workout logic
                    elif self.setup_complete and not self.finished:
                        total_form_errors = 0
                        
                        # 1) POSTURE CHECK (same for both arms)
                        if posture_angle < 160 or posture_angle > 200:
                            self.posture_suggestion = "🧍 Stand straight!"
                            self.posture_color = (0, 0, 255)
                            total_form_errors += 1
                        else:
                            self.posture_suggestion = "✅ Posture good"
                            self.posture_color = (0, 255, 0)

                        # 2) CHECK LEFT ARM
                        if all([l_shoulder, l_elbow, l_wrist]):
                            # Left arm rep counting - STOP at target
                            if self.left_state == "down" and left_elbow_angle < self.UP_THRESHOLD and self.left_reps < self.target_reps:
                                self.left_state = "up"
                                self.left_rep_scored = False
                            elif self.left_state == "up" and left_elbow_angle > self.DOWN_THRESHOLD and self.left_reps < self.target_reps:
                                self.left_state = "down"
                                self.left_reps = min(self.left_reps + 1, self.target_reps)  # Cap at target
                                if not self.left_rep_scored:
                                    self.score += (self.points_per_rep / 2)
                                    self.left_rep_scored = True

                            # Left arm verticality check during curl
                            if self.left_state == "up" and not self.left_rep_scored and self.left_reps < self.target_reps:
                                left_angle, left_vertical = self._check_verticality(l_shoulder, l_elbow)
                                if not left_vertical:
                                    self.left_arm_suggestion = f"✋ Left: {int(left_angle)}°"
                                    self.left_arm_color = (0, 0, 255)
                                    total_form_errors += 1
                                else:
                                    self.left_arm_suggestion = f"✅ Left: {int(left_angle)}°"
                                    self.left_arm_color = (0, 255, 0)

                        # 3) CHECK RIGHT ARM
                        if all([r_shoulder, r_elbow, r_wrist]):
                            # Right arm rep counting - STOP at target
                            if self.right_state == "down" and right_elbow_angle < self.UP_THRESHOLD and self.right_reps < self.target_reps:
                                self.right_state = "up"
                                self.right_rep_scored = False
                            elif self.right_state == "up" and right_elbow_angle > self.DOWN_THRESHOLD and self.right_reps < self.target_reps:
                                self.right_state = "down"
                                self.right_reps = min(self.right_reps + 1, self.target_reps)  # Cap at target
                                if not self.right_rep_scored:
                                    self.score += (self.points_per_rep / 2)
                                    self.right_rep_scored = True

                            # Right arm verticality check during curl
                            if self.right_state == "up" and not self.right_rep_scored and self.right_reps < self.target_reps:
                                right_angle, right_vertical = self._check_verticality(r_shoulder, r_elbow)
                                if not right_vertical:
                                    self.right_arm_suggestion = f"✋ Right: {int(right_angle)}°"
                                    self.right_arm_color = (0, 0, 255)
                                    total_form_errors += 1
                                else:
                                    self.right_arm_suggestion = f"✅ Right: {int(right_angle)}°"
                                    self.right_arm_color = (0, 255, 0)

                        # Apply deductions for form errors
                        if total_form_errors > 0:
                            deduction = (self.points_per_rep / 2) * min(total_form_errors, 2)
                            self.score = max(0.0, self.score - deduction)
                            self.error_count += total_form_errors
                            
                            # Log errors
                            total_reps = max(self.left_reps, self.right_reps)
                            rep_index = min(total_reps + 1, self.target_reps)
                            if rep_index <= self.target_reps and total_form_errors > 0:
                                self.errors_log.append(f"Rep {rep_index}: {total_form_errors} form errors")
                            
                            # Mark reps as scored
                            self.left_rep_scored = True
                            self.right_rep_scored = True

                        # Check if workout finished (when either arm reaches target)
                        max_reps = max(self.left_reps, self.right_reps)
                        if max_reps >= self.target_reps and not self.finished:
                            with self._lock:
                                if not self._workout_saved:
                                    self.finished = True
                                    # Ensure reps are capped
                                    self.left_reps = min(self.left_reps, self.target_reps)
                                    self.right_reps = min(self.right_reps, self.target_reps)
                                    self.score = min(100.0, max(0.0, self.score))
                                    if self.result_queue:
                                        try:
                                            self.result_queue.put({
                                                "reps": max_reps,
                                                "score": round(self.score, 1),
                                                "errors": self.error_count,
                                                "errors_log": self.errors_log,
                                                "finished": True
                                            }, block=False)
                                        except queue.Full:
                                            pass
                                    self._workout_saved = True

                    # Update text cache
                    self._cache.update({
                        "posture_suggestion": self.posture_suggestion,
                        "left_arm_suggestion": self.left_arm_suggestion,
                        "right_arm_suggestion": self.right_arm_suggestion,
                        "posture_color": self.posture_color,
                        "left_arm_color": self.left_arm_color,
                        "right_arm_color": self.right_arm_color,
                    })

            except Exception as e:
                print(f"Error: {e}")

        # Draw HUD
        c = self._cache
        total_reps = max(self.left_reps, self.right_reps)
        
        # Main stats
        try:
            cv2.putText(img, f"Reps: {total_reps}/{self.target_reps}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, f"Score: {round(self.score, 1)}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Errors: {self.error_count}", (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Status messages
            if self.setup_mode and not self.setup_complete:
                if self.setup_start_time:
                    rem = max(0, self.setup_countdown - (time.time() - self.setup_start_time))
                    cv2.putText(img, f"Setup... {rem:.1f}s", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    cv2.putText(img, "Both arms down", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            elif self.finished:
                cv2.putText(img, "🏁 Finished!", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                if self.setup_complete:
                    cv2.putText(img, "✅ Ready! Curl both arms", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Form feedback for both arms
            y_offset = 210
            if c.get("posture_suggestion"):
                cv2.putText(img, c["posture_suggestion"], (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, c.get("posture_color", (0, 255, 0)), 2)
                y_offset += 30
            
            if c.get("left_arm_suggestion"):
                cv2.putText(img, c["left_arm_suggestion"], (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, c.get("left_arm_color", (0, 255, 0)), 2)
                y_offset += 25
            
            if c.get("right_arm_suggestion"):
                cv2.putText(img, c["right_arm_suggestion"], (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, c.get("right_arm_color", (0, 255, 0)), 2)
        except Exception:
            pass

        # Draw landmarks for BOTH arms
        try:
            # Draw left arm (blue)
            if c.get("l_elbow") and c.get("l_shoulder") and c.get("l_wrist"):
                left_color = (0, 255, 0) if self._cache.get("left_arm_color", (0, 255, 0)) == (0, 255, 0) else (0, 0, 255)
                cv2.circle(img, c["l_elbow"], 8, left_color, -1)
                cv2.circle(img, c["l_shoulder"], 6, (255, 0, 0), -1)
                cv2.circle(img, c["l_wrist"], 6, (255, 0, 0), -1)
                cv2.line(img, c["l_shoulder"], c["l_elbow"], (255, 255, 0), 2)
                cv2.line(img, c["l_elbow"], c["l_wrist"], (255, 255, 0), 2)
                cv2.putText(img, f"L:{int(c['left_elbow_angle'])}°", 
                           (c["l_elbow"][0] + 10, c["l_elbow"][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

            # Draw right arm (green)
            if c.get("r_elbow") and c.get("r_shoulder") and c.get("r_wrist"):
                right_color = (0, 255, 0) if self._cache.get("right_arm_color", (0, 255, 0)) == (0, 255, 0) else (0, 0, 255)
                cv2.circle(img, c["r_elbow"], 8, right_color, -1)
                cv2.circle(img, c["r_shoulder"], 6, (0, 255, 0), -1)
                cv2.circle(img, c["r_wrist"], 6, (0, 255, 0), -1)
                cv2.line(img, c["r_shoulder"], c["r_elbow"], (0, 255, 255), 2)
                cv2.line(img, c["r_elbow"], c["r_wrist"], (0, 255, 255), 2)
                cv2.putText(img, f"R:{int(c['right_elbow_angle'])}°", 
                           (c["r_elbow"][0] + 10, c["r_elbow"][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        except Exception:
            pass

        return VideoFrame.from_ndarray(img, format="bgr24")
    
# ----------------------
# Shoulder Press Coach
# ----------------------
class ShoulderPressCoach(VideoTransformerBase):
    """
    Shoulder press coach - uses LEFT arm (appears as right in flipped video)
    """
    def __init__(self, target_reps=10, result_queue=None):
        self.pose = mp_pose.Pose()
        self.reps = 0
        self.state = "waiting"  # waiting -> ready -> up
        self.target_reps = target_reps
        self.result_queue = result_queue
        self._workout_saved = False
        self._lock = threading.Lock()
        self.score = 100
        self.posture_suggestion = ""
        self.arm_suggestion = ""
        self.posture_color = (0, 255, 0)
        self.arm_color = (0, 255, 0)
        self.finished = False
        self.ready_for_count = False
        self.state_locked = False  # Only penalize when locked in a state

            # Add this method RIGHT AFTER the __init__ method:

    def _calculate_vertical_angle(self, shoulder, elbow):
        """
        Calculate how far shoulder-elbow line is from vertical
        Returns: Angle in degrees (0° = perfect vertical, positive = elbow in front, negative = elbow behind)
        """
        try:
            # Calculate vector from shoulder to elbow
            dx = elbow[0] - shoulder[0]  # Horizontal difference
            dy = elbow[1] - shoulder[1]  # Vertical difference
            
            # Avoid division by zero
            if abs(dy) < 1:
                return 90 if dx > 0 else -90
            
            # Calculate angle from vertical: arctan(|dx|/|dy|)
            angle_rad = np.arctan(abs(dx) / abs(dy))
            angle_deg = np.degrees(angle_rad)
            
            # Return signed angle
            if dx > 0:
                return angle_deg  # Elbow is to the right of shoulder
            else:
                return -angle_deg  # Elbow is to the left of shoulder
                
        except Exception:
            return 0  # Default to vertical

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        small_img = cv2.resize(img, (320, 240))
        rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_small)

        self.posture_suggestion = ""
        self.arm_suggestion = ""
        self.posture_color = (0, 255, 0)
        self.arm_color = (0, 255, 0)

        if results.pose_landmarks and not self.finished:
            lm = results.pose_landmarks.landmark
            scale_x = w / 320
            scale_y = h / 240

            def to_original(p):
                return (int(p.x * 320 * scale_x), int(p.y * 240 * scale_y))

            try:
                l_shoulder = to_original(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
                l_elbow = to_original(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
                l_wrist = to_original(lm[mp_pose.PoseLandmark.LEFT_WRIST])
                l_hip = to_original(lm[mp_pose.PoseLandmark.LEFT_HIP])
                l_ankle = to_original(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
            except:
                return VideoFrame.from_ndarray(img, format="bgr24")

            elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            posture_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

            READY_THRESHOLD = 90
            UP_THRESHOLD = 160

           # ----------------------
            # STATE MACHINE
            # ----------------------
            
            if self.state == "waiting":
                if 80 <= elbow_angle <= 100:
                    self.state = "ready"
                    self.ready_for_count = True
                self.arm_suggestion = "📐 Get into 90° starting position"
                if abs(posture_angle - 180) > 10:
                    self.posture_suggestion = "🧍 Straighten your back!"
                    self.posture_color = (0, 0, 255)
                    self.score -= 0.25
                else:
                    self.posture_suggestion = "✅ Good posture!"
                    self.posture_color = (0, 255, 0)

            elif self.state == "ready":
                # Only penalize if user is roughly in ready position
                if self.ready_for_count:
                    if elbow_angle - READY_THRESHOLD < -20:
                        self.arm_suggestion = "📐 Return to 90° position!"
                        self.arm_color = (0, 0, 255)
                        self.score -= 0.25
                    else:
                        self.arm_suggestion = "✅ Ready position good!"
                        self.arm_color = (0, 255, 0)
                    if abs(posture_angle - 180) > 10:
                        self.posture_suggestion = "🧍 Straighten your back!"
                        self.posture_color = (0, 0, 255)
                        self.score -= 0.25
                    else:
                        self.posture_suggestion = "✅ Good posture!"
                        self.posture_color = (0, 255, 0)

                # Move up
                if elbow_angle > UP_THRESHOLD:
                    self.state = "up"

            elif self.state == "up":
                # Only penalize if user is fully extended
                if elbow_angle > UP_THRESHOLD - 5:  # small tolerance
                    if elbow_angle < UP_THRESHOLD:
                        self.arm_suggestion = "⬆️ Extend arms fully!"
                        self.arm_color = (0, 0, 255)
                    else:
                        self.arm_suggestion = "✅ Full extension!"
                        self.arm_color = (0, 255, 0)
                    if abs(posture_angle - 180) > 10:
                        self.posture_suggestion = "🧍 Straighten your back!"
                        self.posture_color = (0, 0, 255)
                    else:
                        self.posture_suggestion = "✅ Good posture!"
                        self.posture_color = (0, 255, 0)

                # Detect rep when coming back to 90° position
                if 80 <= elbow_angle <= 100 and self.ready_for_count:
                    self.reps += 1
                    self.state = "waiting"  # <-- changed from ready to waiting

            # Check if finished
            if self.reps >= self.target_reps and not self._workout_saved:
                with self._lock:
                    if not self._workout_saved:
                        self.finished = True
                        self._workout_saved = True
                        if self.result_queue:
                            try:
                                self.result_queue.put({
                                    "reps": self.reps,
                                    "score": max(0, self.score),
                                    "finished": True
                                }, block=False)
                            except queue.Full:
                                pass


            # Draw HUD
            cv2.putText(img, f"Reps: {self.reps}/{self.target_reps}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, f"Score: {int(self.score)}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, self.posture_suggestion, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.posture_color, 2)
            cv2.putText(img, self.arm_suggestion, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.arm_color, 2)

            # Draw elbow marker and angle
            angle_color = (0, 255, 0) if (80 <= elbow_angle <= 100 or elbow_angle > UP_THRESHOLD) else (0, 0, 255)
            cv2.circle(img, l_elbow, 8, angle_color, -1)
            cv2.putText(img, f"{int(elbow_angle)}°", (l_elbow[0] + 10, l_elbow[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)

        return VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------
# Main App
# ----------------------
def main_app():
    tabs = ["Workouts", "Stats", "Logout"]
    choice = st.sidebar.selectbox("Navigation", tabs)

    if choice == "Workouts":
        st.subheader("💪 Workouts")

        # Choose exercise
        exercise_type = st.selectbox("Choose Exercise", ["Squats", "Bicep Curls", "Shoulder Press"])

        st.session_state.target_reps = st.number_input(
            "Enter number of reps",
            min_value=1,
            max_value=50,
            value=st.session_state.target_reps
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Workout"):
                st.session_state.start_squats = True
                # Clear the queue
                while not st.session_state.workout_result_queue.empty():
                    try:
                        st.session_state.workout_result_queue.get_nowait()
                    except queue.Empty:
                        break
                st.session_state._current_exercise = exercise_type
                st.rerun()

        with col2:
            if st.button("Stop Workout"):
                st.session_state.start_squats = False
                st.rerun()

        if st.session_state.start_squats:
            target_reps_value = st.session_state.target_reps
            result_queue = st.session_state.workout_result_queue
            chosen_ex = st.session_state.get("_current_exercise", "Squats")

            # --- Updated processor selection ---
            if chosen_ex == "Squats":
                processor = PoseCoach
            elif chosen_ex == "Bicep Curls":
                processor = BicepCurlCoach
            else:  # Shoulder Press
                processor = ShoulderPressCoach

            webrtc_ctx = webrtc_streamer(
                key=f"exercise-session-{chosen_ex}",
                video_processor_factory=lambda: processor(
                    target_reps=target_reps_value,
                    result_queue=result_queue
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                mode=WebRtcMode.SENDRECV,
            )

            # Check for completed workout
            try:
                result = result_queue.get_nowait()
                if result.get("finished"):
                    workout_record = {
                        "user": st.session_state.current_user,
                        "exercise": chosen_ex.lower(),
                        "reps": result["reps"],
                        "score": result["score"],
                        "timestamp": pd.Timestamp.now()
                    }
                    st.session_state.workout_data.append(workout_record)
                    save_workout_data(st.session_state.workout_data)

                    st.success(f"🎉 Workout completed! {chosen_ex} — Reps: {result['reps']}, Score: {result['score']}")
                    st.balloons()

                    if st.button("Start New Workout"):
                        st.session_state.start_squats = False
                        st.rerun()
            except queue.Empty:
                pass
        else:
            st.info("Click 'Start Workout' to begin your workout!")


    elif choice == "Stats":
        st.subheader("📊 Stats")
        data = st.session_state.workout_data

        if data:
            df = pd.DataFrame(data)
            user_df = df[df["user"] == st.session_state.current_user]

            if not user_df.empty:

                # Convert timestamps to datetime
                user_df["timestamp"] = pd.to_datetime(user_df["timestamp"])
                user_df["date"] = user_df["timestamp"].dt.date
                user_df["hour"] = user_df["timestamp"].dt.hour
                user_df["day_name"] = user_df["timestamp"].dt.day_name()

                # ===========================
                # ==== TOP: RAW DATA TABLE ===
                # ===========================
                st.dataframe(user_df.style.highlight_max(axis=0), height=220)

                # ===========================
                # === KPI METRIC CARDS ======
                # ===========================
                total_workouts = len(user_df)
                total_reps = int(user_df["reps"].sum())
                avg_score = round(user_df["score"].mean(), 1)

                c1, c2, c3 = st.columns(3)
                c1.metric("🟦 Total Workouts", total_workouts)
                c2.metric("🟩 Total Reps", total_reps)
                c3.metric("⭐ Average Score", avg_score)

                st.markdown("---")

                # ===========================
                # == DAILY SUMMARY CHART ====
                # ===========================
                # prepare daily
                daily = user_df.groupby("date").agg({"reps": "sum", "score": "mean"}).reset_index()

                # melt to long format
                daily_melt = daily.melt(id_vars="date", value_vars=["reps", "score"],
                                        var_name="metric", value_name="value")

                color_scale = alt.Scale(domain=["reps", "score"], range=["#5DADE2", "#58D68D"])

                grouped = (
                    alt.Chart(daily_melt)
                    .mark_bar()
                    .encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-45)),
                        xOffset="metric:N",                         # <--- side-by-side grouping
                        y=alt.Y("value:Q", title="Value"),
                        color=alt.Color("metric:N", scale=color_scale, title="Metric"),
                        tooltip=[alt.Tooltip("date:T", title="Date"),
                                alt.Tooltip("metric:N", title="Metric"),
                                alt.Tooltip("value:Q", title="Value")]
                    )
                    .properties(height=360)
                )

                st.altair_chart(grouped, use_container_width=True)



                # ===========================
                # ==== DOUGHNUT CHART =======
                # ===========================
                st.subheader("🍩 Exercise Distribution")

                # Count how many workouts for each exercise
                exercise_counts = (
                    user_df.groupby("exercise")["exercise"].count()
                    .reset_index(name="count")
                )

                # Compute proportions
                exercise_counts["percentage"] = (
                    exercise_counts["count"] / exercise_counts["count"].sum()
                ) * 100

                # Create doughnut chart (Altair)
                doughnut = alt.Chart(exercise_counts).mark_arc(innerRadius=70).encode(
                    theta=alt.Theta(field="count", type="quantitative"),
                    color=alt.Color(field="exercise", type="nominal"),
                    tooltip=[
                        alt.Tooltip("exercise:N", title="Exercise"),
                        alt.Tooltip("count:Q", title="Sessions"),
                        alt.Tooltip("percentage:Q", format=".1f", title="% of Total")
                    ]
                ).properties(
                    width=350,
                    height=350
                )

                st.altair_chart(doughnut, use_container_width=False)

                # ===========================
                # === SCORE HEATMAP =========
                # ===========================
                st.subheader("🔥 Score Heatmap by Day")

                heatmap = alt.Chart(user_df).mark_rect().encode(
                    x=alt.X("hour:O", title="Hour of Day"),
                    y=alt.Y("day_name:O", title="Day"),
                    color=alt.Color("score:Q", scale=alt.Scale(scheme="yelloworangebrown")),
                    tooltip=["timestamp", "score"]
                ).properties(height=250)

                st.altair_chart(heatmap, use_container_width=True)

                # ===========================
                # ====== SCATTER PLOT =======
                # ===========================
                st.subheader("⚖️ Form Quality vs. Rep Volume")

                scatter = alt.Chart(user_df).mark_circle(size=200).encode(
                    x="reps:Q",
                    y="score:Q",
                    color="exercise:N",
                    tooltip=["exercise", "reps", "score"]
                ).properties(height=300)

                st.altair_chart(scatter, use_container_width=True)

            else:
                st.info("No workout data yet. Complete a workout to see stats here!")
        else:
            st.info("No workout data yet. Complete a workout to see stats here!")

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.start_squats = False
        st.success("Logged out successfully!")
        st.rerun()

# ----------------------
# Run App
# ----------------------
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()