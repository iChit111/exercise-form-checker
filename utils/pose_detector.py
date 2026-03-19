"""
utils/pose_detector.py — MediaPipe Pose Detection Wrapper
==========================================================
UPDATED for MediaPipe 0.10.30+
The old mp.solutions.pose API was removed. This file uses the new Tasks API.

YOUR ROLE: Pose Detection (Person 2)

What this file does:
    1. Initializes MediaPipe PoseLandmarker (the new AI model)
    2. Detects body landmarks (keypoints) in each webcam frame
    3. Draws the skeleton overlay on the frame
    4. Returns landmark data in a format the exercise modules can use

How MediaPipe Landmarks Work:
-----------------------------
MediaPipe detects 33 points on the human body. Each landmark has:
    x          — Left/right position (0.0 = left edge, 1.0 = right edge)
    y          — Up/down position   (0.0 = top edge,  1.0 = bottom edge)
    z          — Depth (less reliable)
    visibility — Confidence score (0.0–1.0)

Important landmark indices:
    LEFT_SHOULDER=11   RIGHT_SHOULDER=12
    LEFT_ELBOW=13      RIGHT_ELBOW=14
    LEFT_WRIST=15      RIGHT_WRIST=16
    LEFT_HIP=23        RIGHT_HIP=24
    LEFT_KNEE=25       RIGHT_KNEE=26
    LEFT_ANKLE=27      RIGHT_ANKLE=28
"""

import cv2
import numpy as np
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)


# ==============================================================================
# LANDMARK INDEX CONSTANTS
# Same as before — use these in exercise files for readable code
# ==============================================================================

class LM:
    """Shorthand constants for MediaPipe's 33 pose landmark indices."""
    NOSE            = 0
    LEFT_EYE_INNER  = 1
    LEFT_EYE        = 2
    LEFT_EYE_OUTER  = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE       = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR        = 7
    RIGHT_EAR       = 8
    MOUTH_LEFT      = 9
    MOUTH_RIGHT     = 10
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_PINKY      = 17
    RIGHT_PINKY     = 18
    LEFT_INDEX      = 19
    RIGHT_INDEX     = 20
    LEFT_THUMB      = 21
    RIGHT_THUMB     = 22
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# ==============================================================================
# MODEL DOWNLOAD HELPER
# The new Tasks API requires a .task model file downloaded separately.
# This function downloads it automatically if it's not already present.
# ==============================================================================

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

def ensure_model_exists():
    """
    Downloads the MediaPipe pose model file if it doesn't already exist.
    Only needs to run once — after that the file is cached locally.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[PoseDetector] Downloading pose model to '{MODEL_PATH}'...")
        print("[PoseDetector] This only happens once. Please wait...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[PoseDetector] Model downloaded successfully!")
    else:
        print(f"[PoseDetector] Model file found: '{MODEL_PATH}'")


# ==============================================================================
# THE MAIN POSEDETECTOR CLASS
# Same interface as before — .detect(frame) still returns a list of 33 landmarks
# ==============================================================================

class PoseDetector:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API) for easy use by the team.

    Usage — identical to the old version:
        detector = PoseDetector()
        landmarks = detector.detect(frame)
        if landmarks:
            hip = landmarks[LM.LEFT_HIP]
            print(hip.x, hip.y)
    """

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Sets up the MediaPipe PoseLandmarker model.

        Args:
            detection_confidence (float): Min confidence to detect a person (0.0–1.0)
            tracking_confidence  (float): Min confidence to keep tracking  (0.0–1.0)
        """
        print("[PoseDetector] Initializing MediaPipe PoseLandmarker (Tasks API)...")

        # Download model file if needed
        ensure_model_exists()

        # --- Configure the PoseLandmarker ---
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

        options = PoseLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode is used for webcam streams — it tracks across frames
            # (faster and smoother than IMAGE mode which treats each frame independently)
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            num_poses=1,            # We only need to track one person
            output_segmentation_masks=False
        )

        # Create the landmarker instance
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Store drawing utilities for skeleton overlay
        self.mp_drawing       = vision.drawing_utils
        self.mp_drawing_styles = vision.drawing_styles

        # Frame counter — VIDEO mode requires a timestamp per frame
        self._frame_index = 0

        print("[PoseDetector] Ready!")

    def detect(self, frame, draw_skeleton=True):
        """
        Analyzes a webcam frame and returns detected body landmarks.

        Args:
            frame (numpy.ndarray): BGR image from OpenCV.
            draw_skeleton (bool): If True, draws skeleton on the frame in-place.

        Returns:
            list or None:
                - List of 33 landmark objects if a pose is detected.
                  Each has .x, .y, .z (0.0–1.0) and .visibility.
                - None if no pose detected.
        """

        # --- Convert BGR (OpenCV) → RGB (MediaPipe) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Wrap in MediaPipe Image object ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- Calculate timestamp in milliseconds ---
        # VIDEO mode requires an increasing timestamp for each frame
        timestamp_ms = self._frame_index * 33  # ~30fps = 33ms per frame
        self._frame_index += 1

        # --- Run pose detection ---
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- Check if any pose was detected ---
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # --- Get the first person's landmarks ---
        # result.pose_landmarks is a list of poses (we only track 1 person)
        landmarks = result.pose_landmarks[0]  # List of 33 NormalizedLandmark objects

        # --- Draw skeleton if requested ---
        if draw_skeleton:
            self._draw_skeleton(frame, result)

        return landmarks

    def _draw_skeleton(self, frame, result):
        """
        Draws pose skeleton onto the frame in-place.
        Uses OpenCV directly to draw dots and lines — no mediapipe.framework needed.
        """
        if not result.pose_landmarks:
            return

        frame_h, frame_w = frame.shape[:2]

        # MediaPipe's 33-point body connections (pairs of landmark indices)
        CONNECTIONS = [
            (11,12),(11,13),(13,15),(12,14),(14,16),  # Arms
            (11,23),(12,24),(23,24),                   # Torso
            (23,25),(25,27),(27,29),(27,31),           # Left leg
            (24,26),(26,28),(28,30),(28,32),           # Right leg
            (0,1),(1,2),(2,3),(3,7),                   # Face left
            (0,4),(4,5),(5,6),(6,8),                   # Face right
            (9,10),(15,17),(15,19),(15,21),            # Mouth, left hand
            (16,18),(16,20),(16,22),                   # Right hand
        ]

        for pose_landmarks in result.pose_landmarks:
            # Convert normalized coords to pixel coords
            points = []
            for lm in pose_landmarks:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                points.append((px, py))

            # Draw connection lines first (so dots appear on top)
            for start_idx, end_idx in CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx],
                             (200, 200, 200), 2)  # Light gray lines

            # Draw landmark dots
            for px, py in points:
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)  # Green dots

    def get_pixel_coordinates(self, landmark, frame_width, frame_height):
        """
        Converts normalized landmark coordinates (0.0–1.0) to pixel coordinates.

        Args:
            landmark: A single landmark object.
            frame_width (int):  Frame width in pixels.
            frame_height (int): Frame height in pixels.

        Returns:
            tuple: (x_pixels, y_pixels)
        """
        return (int(landmark.x * frame_width), int(landmark.y * frame_height))

    def is_landmark_visible(self, landmark, threshold=0.5):
        """
        Checks if a landmark is reliably visible.

        Returns:
            bool: True if visibility >= threshold.
        """
        return landmark.visibility >= threshold

    def close(self):
        """Releases MediaPipe resources. Call when the app shuts down."""
        self.landmarker.close()
        print("[PoseDetector] Resources released.")


# ==============================================================================
# STANDALONE TEST
# Run: py -3.11 utils/pose_detector.py
# You should see a webcam window with a skeleton. Press Q to quit.
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PoseDetector Standalone Test (MediaPipe 0.10.30+)")
    print("  Stand in front of your webcam and move around!")
    print("  Press Q to quit.")
    print("=" * 60)

    detector = PoseDetector(detection_confidence=0.5, tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[TEST] Webcam opened. Starting detection loop...")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        landmarks = detector.detect(frame, draw_skeleton=True)

        frame_h, frame_w, _ = frame.shape

        if landmarks is not None:
            cv2.putText(frame, "Pose Detected!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)

            debug_landmarks = {
                "L.Shoulder": LM.LEFT_SHOULDER,
                "L.Hip":      LM.LEFT_HIP,
                "L.Knee":     LM.LEFT_KNEE,
                "L.Ankle":    LM.LEFT_ANKLE,
            }

            y_offset = 100
            for name, index in debug_landmarks.items():
                lm = landmarks[index]
                px, py = detector.get_pixel_coordinates(lm, frame_w, frame_h)
                visible = "✓" if detector.is_landmark_visible(lm) else "✗"
                text = f"{name}: ({px}, {py})  vis={lm.visibility:.2f} {visible}"
                cv2.putText(frame, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 35
        else:
            cv2.putText(frame, "No pose detected — step into view!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.putText(frame, "Press Q to quit", (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow("PoseDetector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[TEST] Done!")