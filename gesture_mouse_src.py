# src/gesture_classifier.py
# Classifies hand gestures from MediaPipe landmarks
# Author: Narayana Vasan S S

from hand_detector import HandDetector


class GestureClassifier:
    """Maps landmark positions to gesture names."""

    CLICK_THRESHOLD  = 35   # pixels — pinch distance to trigger click
    SCROLL_THRESHOLD = 40   # pixels — pinch distance for scroll

    def classify(self, landmarks):
        """
        Returns a dict: {
            'gesture': str,
            'index_tip': (x, y),
            'fingers_up': [bool x5]
        }
        """
        if not landmarks:
            return None

        lm = landmarks[0]  # First hand only

        # Which fingers are extended (tip y < pip y for index–pinky; thumb uses x)
        fingers_up = [
            lm[HandDetector.THUMB_TIP][0]  < lm[HandDetector.THUMB_TIP - 1][0],   # Thumb (x-axis)
            lm[HandDetector.INDEX_TIP][1]  < lm[HandDetector.INDEX_MCP][1],        # Index
            lm[HandDetector.MIDDLE_TIP][1] < lm[HandDetector.MIDDLE_MCP][1],       # Middle
            lm[HandDetector.RING_TIP][1]   < lm[HandDetector.INDEX_MCP][1] + 10,   # Ring (approx)
            lm[HandDetector.PINKY_TIP][1]  < lm[HandDetector.INDEX_MCP][1] + 10,   # Pinky (approx)
        ]

        thumb_to_index  = HandDetector.distance(lm[HandDetector.THUMB_TIP], lm[HandDetector.INDEX_TIP])
        thumb_to_middle = HandDetector.distance(lm[HandDetector.THUMB_TIP], lm[HandDetector.MIDDLE_TIP])

        # Gesture logic
        gesture = 'none'

        if fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            gesture = 'move'  # Only index up → move cursor

        elif fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            gesture = 'scroll'  # Index + middle up → scroll

        elif thumb_to_index < self.CLICK_THRESHOLD:
            gesture = 'left_click'  # Thumb–index pinch

        elif thumb_to_middle < self.SCROLL_THRESHOLD:
            gesture = 'right_click'  # Thumb–middle pinch

        elif not any(fingers_up[1:]):
            gesture = 'fist'  # All fingers down → pause

        return {
            'gesture':    gesture,
            'index_tip':  lm[HandDetector.INDEX_TIP],
            'fingers_up': fingers_up,
            'thumb_index_dist':  round(thumb_to_index, 1),
            'thumb_middle_dist': round(thumb_to_middle, 1),
        }


# ──────────────────────────────────────────────────────────────────
# src/mouse_controller.py
# ──────────────────────────────────────────────────────────────────
# OS-level mouse event dispatcher
# Author: Narayana Vasan S S

import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0


class MouseController:
    """Maps camera coordinates to screen coordinates and fires OS mouse events."""

    DEAD_ZONE   = 10    # pixels — ignore tiny movements to reduce jitter
    SMOOTH      = 0.18  # interpolation factor (lower = smoother but laggier)
    CLICK_DELAY = 0.5   # seconds between clicks to prevent accidental repeats

    def __init__(self, cam_w, cam_h):
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w    = cam_w
        self.cam_h    = cam_h
        self.prev_x   = self.screen_w // 2
        self.prev_y   = self.screen_h // 2
        self._last_click_time = 0

    def _cam_to_screen(self, cx, cy):
        """Map camera pixel → screen pixel with numpy interpolation."""
        sx = int(np.interp(cx, [0, self.cam_w], [0, self.screen_w]))
        sy = int(np.interp(cy, [0, self.cam_h], [0, self.screen_h]))
        return sx, sy

    def move(self, cx, cy):
        sx, sy = self._cam_to_screen(cx, cy)
        dx, dy = abs(sx - self.prev_x), abs(sy - self.prev_y)
        if dx < self.DEAD_ZONE and dy < self.DEAD_ZONE:
            return  # In dead zone — ignore
        # Smooth interpolation
        smooth_x = int(self.prev_x + (sx - self.prev_x) * self.SMOOTH * 6)
        smooth_y = int(self.prev_y + (sy - self.prev_y) * self.SMOOTH * 6)
        pyautogui.moveTo(smooth_x, smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def _can_click(self):
        now = time.time()
        if now - self._last_click_time >= self.CLICK_DELAY:
            self._last_click_time = now
            return True
        return False

    def left_click(self):
        if self._can_click():
            pyautogui.click()

    def right_click(self):
        if self._can_click():
            pyautogui.rightClick()

    def scroll(self, direction=1):
        """direction: 1=up, -1=down"""
        pyautogui.scroll(direction * 3)


# ──────────────────────────────────────────────────────────────────
# src/main.py
# ──────────────────────────────────────────────────────────────────
# Entry point — Virtual Mouse Application
# Author: Narayana Vasan S S
# Run: python src/main.py

import cv2
import time
from hand_detector      import HandDetector
from gesture_classifier import GestureClassifier
from mouse_controller   import MouseController

CAM_INDEX = 0
CAM_W, CAM_H = 640, 480


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector   = HandDetector()
    classifier = GestureClassifier()
    mouse      = MouseController(CAM_W, CAM_H)

    prev_time = 0
    print("✋ Virtual Mouse active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel
        frame, landmarks = detector.detect(frame)
        result = classifier.classify(landmarks)

        if result:
            gesture   = result['gesture']
            idx_tip   = result['index_tip']

            if gesture == 'move':
                mouse.move(idx_tip[0], idx_tip[1])
            elif gesture == 'left_click':
                mouse.left_click()
                cv2.putText(frame, 'LEFT CLICK', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            elif gesture == 'right_click':
                mouse.right_click()
                cv2.putText(frame, 'RIGHT CLICK', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            elif gesture == 'scroll':
                mouse.scroll(1)
                cv2.putText(frame, 'SCROLL', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)

            # Draw gesture label
            cv2.putText(frame, f'Gesture: {gesture}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS counter
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 1e-9))
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps}', (CAM_W - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Virtual Mouse — Narayana Vasan', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Virtual Mouse stopped.")


if __name__ == '__main__':
    main()
