# gesture-virtual-mouse
Real-time hand gesture mouse control with MediaPipe
# 🖐️ Real-Time Hand Gesture Recognition — Virtual Mouse

Control your computer mouse using only your webcam and hand gestures. No extra hardware needed.

## ✨ Features
- Real-time 21-point hand landmark detection (MediaPipe)
- Mouse movement via index fingertip coordinates
- Left click: thumb + index finger pinch
- Right click: thumb + middle finger pinch
- Scroll: two-finger gesture
- Dead-zone threshold to eliminate cursor jitter
- 0.5s cooldown to prevent accidental clicks
- 30 FPS real-time performance

## 🛠️ Tech Stack
`Python 3.10+` `OpenCV` `MediaPipe` `PyAutoGUI` `NumPy`

## 📁 Project Structure
```
gesture-mouse/
├── src/
│   ├── hand_detector.py    ← MediaPipe landmark detection
│   ├── gesture_classifier.py ← Gesture recognition logic
│   ├── mouse_controller.py ← OS mouse event dispatcher
│   └── main.py             ← Entry point
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Run
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/gesture-mouse.git
cd gesture-mouse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python src/main.py
```

## 🎮 Controls
| Gesture | Action |
|---------|--------|
| Index finger up | Move cursor |
| Thumb + Index pinch | Left click |
| Thumb + Middle pinch | Right click |
| Index + Middle up | Scroll mode |
| Fist | Stop / pause |

## 📋 Requirements
```
opencv-python>=4.8.0
mediapipe>=0.10.0
pyautogui>=0.9.54
numpy>=1.24.0
```

## 👤 Author
**Narayana Vasan S S** — Third Year Project, SASTRA Deemed University
