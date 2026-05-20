# src/hand_detector.py
# MediaPipe Hand Landmark Detection Module
# Author: Narayana Vasan S S

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Detects hand landmarks using MediaPipe."""

    # Landmark indices
    WRIST          = 0
    THUMB_TIP      = 4
    INDEX_TIP      = 8
    MIDDLE_TIP     = 12
    RING_TIP       = 16
    PINKY_TIP      = 20
    INDEX_MCP      = 5
    MIDDLE_MCP     = 9

    def __init__(self, max_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        self.mp_hands   = mp.solutions.hands
        self.mp_draw    = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.hands      = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame_bgr):
        """Detect hands in a BGR frame. Returns (annotated_frame, landmarks_list)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        all_landmarks = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
                h, w, _ = frame.shape
                lms = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark]
                all_landmarks.append(lms)

        return frame, all_landmarks

    @staticmethod
    def distance(p1, p2):
        """Euclidean distance between two (x, y) points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
