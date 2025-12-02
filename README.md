# 🏋️‍♂️ IskoCoach — Real-Time Squat Pose Coach

IskoCoach is a real-time computer vision fitness assistant built using **MediaPipe Pose** and **Streamlit**.  
It counts **squat repetitions**, evaluates **form quality**, gives **live feedback**, and maintains each user’s workout history with data visualizations.

🎯 **Goal:** Help beginners learn proper squat form with AI-powered guidance — no wearable sensors needed!

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 👤 User Accounts | Login & registration with local JSON storage |
| 📸 Web-based Pose Tracking | Webcam input with MediaPipe Pose |
| 🔄 Repetition Counter | Knee-angle-based squat detection |
| 📏 Form Feedback | Detects knee overextension & improper back posture |
| 🧮 Scoring System | Deducts points for form mistakes during reps |
| 💾 Workout Logging | Saves reps, score, timestamp & user |
| 📊 Stats Dashboard | Interactive Altair charts (history & performance) |

---

## 🧠 Computer Vision Model

- **Model Used:** MediaPipe BlazePose (pretrained)
- **What It Tracks:** 2D skeletal keypoints (33 body landmarks)
- **How It Works (Simplified):**
  1. Extract hip → knee → ankle joint coordinates
  2. Compute **knee angle** per frame
  3. Detect **Up → Down → Up** transitions using threshold rules
  4. Track spine alignment using vector geometry (dot-product)

### 🔍 Limitations
- Works best with clear full-body visibility
- Requires upright front/side camera angle for accurate angles
- Heuristic-based form evaluation → can improve with ML classifier in future

---

## 🧪 Evaluation

We evaluated our rep counter and scoring system using:
- **Ground-truth manual rep annotations**
- Per-frame logs from webcam sessions

Metrics included:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **% accuracy within ±1 rep**
- **Correlation between form score and rep error**

📊 Charts generated:
- Error histogram  
- Score vs Absolute Error  
- Reps over time per user  
- User comparison plots  

🎯 Summary: The system performs accurately under controlled conditions, with only minor miscounts when the user is partially occluded or camera height varies.

👉 Evaluation scripts are in the `evaluation/` folder.

---

## 🖥️ User Interface Preview



---

## 🏗️ Project Structure

```
.
├─ app.py               # Streamlit app
├─ users.json           # Saved user accounts
├─ workout_data.json    # Workout logs (auto-generated)
├─ session_logs/        # Optional per-frame rep logs
├─ evaluation/
│  ├─ evaluate.py       # Metrics & graph generation
│  ├─ annotate.py       # Ground-truth labeling tool
│  └─ reports/          # Exported charts & csvs
├─ assets/              # Images for README or presentation
├─ requirements.txt
└─ README.md
```

---

## 🧩 Installation & Running

### 1️⃣ Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2️⃣ Run the app

```bash
streamlit run app.py
```

Once started, visit the displayed URL (typically `http://localhost:8501`) and allow webcam access.

---

## 📈 Stats and Progress Tracking

Available under **Sidebar → Stats**:
- Interactive bar chart of reps 🏋️‍♀️
- Color encodes performance score
- Total reps completed
- Average score
- Total workouts completed

---

## ✨ Future Improvements

| Upgrade | Benefit |
|--------|---------|
| Multi-camera support | Better 3D form estimation |
| Learnable posture classifier | More reliable back straightness detection |
| Personalized calibration step | Adapts to user leg proportions & camera height |
| Exercise expansion (push-ups, lunges…) | Full AI workout platform capability |
| Cloud database login | Multi-device progress sync |

---

## 👥 Team Members

- Mariano, Isaiah John
- Montealto, Meluisa
- Regalario, Jeremiah Daniel

---

## 🙌 Tools used
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) Pose by Google Research  
- Streamlit-WebRTC for real-time webcam support  
- Altair for interactive visualization  

---

> _From UP students, for healthier Iskos & Iskas!_  
**Let AI Coach You — One Squat at a Time. 💪🔥**
