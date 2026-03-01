"""
utils/pose_detector.py — MediaPipe Pose Detection Wrapper
==========================================================
YOUR ROLE: Pose Detection
This file is your responsibility! It wraps Google's MediaPipe Pose library
into a clean, easy-to-use class that the rest of the team can import.

What this file does:
    1. Initializes MediaPipe Pose (the AI model that finds body keypoints)
    2. Detects body landmarks (keypoints) in each webcam frame
    3. Draws the skeleton overlay on the frame so the user can see it
    4. Returns landmark data in a format the exercise modules can use

How MediaPipe Landmarks Work:
-----------------------------
MediaPipe Pose detects 33 specific points on the human body, called LANDMARKS.
Each landmark represents a body part (like your left knee or right shoulder).

Every landmark has three coordinates:
    x  — How far LEFT or RIGHT the point is (0.0 = left edge, 1.0 = right edge)
    y  — How far UP or DOWN the point is   (0.0 = top edge,  1.0 = bottom edge)
    z  — Depth (how far toward/away from the camera — less reliable, use carefully)
    visibility — How confident MediaPipe is that this point is visible (0.0–1.0)

These x/y values are NORMALIZED, meaning they're percentages of the frame size,
not pixel coordinates. To get pixel coordinates, multiply by the frame width/height.

The 33 landmarks are numbered 0–32. Here are the most important ones for exercise analysis:

    NOSE            = 0
    LEFT_EYE        = 1      RIGHT_EYE         = 4
    LEFT_EAR        = 7      RIGHT_EAR         = 8
    LEFT_SHOULDER   = 11     RIGHT_SHOULDER    = 12
    LEFT_ELBOW      = 13     RIGHT_ELBOW       = 14
    LEFT_WRIST      = 15     RIGHT_WRIST       = 16
    LEFT_HIP        = 23     RIGHT_HIP         = 24
    LEFT_KNEE       = 25     RIGHT_KNEE        = 26
    LEFT_ANKLE      = 27     RIGHT_ANKLE       = 28
    LEFT_HEEL       = 29     RIGHT_HEEL        = 30
    LEFT_FOOT_INDEX = 31     RIGHT_FOOT_INDEX  = 32

For exercise analysis, you'll mostly care about:
    - Squats:      Hips (23/24), Knees (25/26), Ankles (27/28), Shoulders (11/12)
    - Push-ups:    Shoulders (11/12), Elbows (13/14), Wrists (15/16), Hips (23/24)
    - Bicep curls: Shoulders (11/12), Elbows (13/14), Wrists (15/16)
"""

import cv2          # OpenCV — for drawing on frames and color conversion
import mediapipe as mp  # MediaPipe — Google's AI pose detection library
import numpy as np  # NumPy — for coordinate math


# ==============================================================================
# MEDIAPIPE LANDMARK INDEX CONSTANTS
# Using named constants instead of "magic numbers" makes the code readable.
# Instead of writing landmarks[23], you can write landmarks[LM.LEFT_HIP]
# ==============================================================================

class LM:
    """
    Shorthand constants for MediaPipe's 33 pose landmark indices.
    Use these when accessing landmarks so your code is self-documenting.

    Example:
        hip_landmark = landmarks[LM.LEFT_HIP]
    """
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
# THE MAIN POSEDETECTOR CLASS
# ==============================================================================

class PoseDetector:
    """
    A wrapper around MediaPipe Pose that makes it easy to detect body landmarks
    in webcam frames and draw a skeleton overlay on top.

    Usage (how your teammates will use this):
        detector = PoseDetector()
        landmarks = detector.detect(frame)   # Pass in a webcam frame
        if landmarks:
            hip = landmarks[LM.LEFT_HIP]
            print(hip.x, hip.y)  # Normalized coordinates (0.0–1.0)
    """

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Sets up the MediaPipe Pose model.

        Args:
            detection_confidence (float): How confident MediaPipe must be to say
                "yes, I see a person here." Range: 0.0–1.0. Higher = more strict.
                0.5 is a good starting point. Lower it if detection is spotty.

            tracking_confidence (float): Once a person is found, how confident
                MediaPipe must be to keep tracking them frame-to-frame.
                Range: 0.0–1.0. Lower = smoother but might track wrong points.
        """
        print("[PoseDetector] Initializing MediaPipe Pose...")

        # mp.solutions.pose gives us access to the Pose model
        self.mp_pose = mp.solutions.pose

        # mp.solutions.drawing_utils lets us draw the skeleton lines and dots
        self.mp_drawing = mp.solutions.drawing_utils

        # mp.solutions.drawing_styles gives us pre-made colors for the skeleton
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create the actual Pose model instance
        # static_image_mode=False means we're processing a VIDEO (not single images)
        # This lets MediaPipe track poses across frames, which is faster and smoother
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0=fastest/least accurate, 2=slowest/most accurate
            smooth_landmarks=True,       # Smooths jitter between frames — keep this True
            enable_segmentation=False,   # We don't need body segmentation for this app
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        print("[PoseDetector] Ready!")

    def detect(self, frame, draw_skeleton=True):
        """
        Analyzes a single webcam frame and returns the detected body landmarks.

        This is the main method your teammates will call every frame.

        Args:
            frame (numpy.ndarray): A BGR image from OpenCV (what cv2.VideoCapture gives you).
                                   Shape is (height, width, 3).
            draw_skeleton (bool): If True, draws the pose skeleton directly onto the frame.
                                  The frame is modified IN PLACE, so the caller sees the
                                  skeleton without needing to do anything extra.

        Returns:
            list or None:
                - If a pose is detected: a list of 33 landmark objects.
                  Each landmark has .x, .y, .z (normalized 0.0–1.0) and .visibility.
                  Access like: landmarks[LM.LEFT_KNEE].x
                - If no pose is detected: None
                  (Always check for None before using landmarks!)

        Example:
            landmarks = detector.detect(frame)
            if landmarks is not None:
                knee = landmarks[LM.LEFT_KNEE]
                print(f"Left knee is at {knee.x:.2f}, {knee.y:.2f}")
        """

        # --- Step 1: Convert color space ---
        # OpenCV loads frames in BGR (Blue-Green-Red) order.
        # MediaPipe expects RGB (Red-Green-Blue) order.
        # We MUST convert, otherwise colors will look wrong and detection suffers.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Step 2: Optional performance trick ---
        # Marking the image as not writeable before processing can speed things up
        # slightly because MediaPipe doesn't need to worry about the array changing.
        rgb_frame.flags.writeable = False

        # --- Step 3: Run pose detection ---
        # This is where the AI magic happens! MediaPipe analyzes the frame and
        # finds all 33 body landmark positions.
        results = self.pose.process(rgb_frame)

        # --- Step 4: Re-enable writing so we can draw on the frame ---
        rgb_frame.flags.writeable = True

        # --- Step 5: Check if a pose was actually detected ---
        # results.pose_landmarks is None if no person was found in the frame
        if results.pose_landmarks is None:
            # No person detected — return None so the caller knows
            return None

        # --- Step 6: Draw the skeleton overlay (optional) ---
        if draw_skeleton:
            self._draw_skeleton(frame, results)

        # --- Step 7: Return the landmark list ---
        # results.pose_landmarks.landmark is a list of 33 NormalizedLandmark objects.
        # Each one has .x, .y, .z (all 0.0–1.0) and .visibility (0.0–1.0).
        return results.pose_landmarks.landmark

    def _draw_skeleton(self, frame, results):
        """
        Draws the pose skeleton (dots + connecting lines) onto the frame.
        This modifies the frame IN PLACE — no need to return it.

        This is a "private" method (notice the underscore prefix). It's meant to be
        called only from inside this class, not by your teammates directly.

        Args:
            frame (numpy.ndarray): The BGR frame to draw on.
            results: The raw MediaPipe results object from self.pose.process().
        """
        # mp_drawing.draw_landmarks() handles all the drawing for us.
        # It draws dots at each landmark and lines connecting related landmarks.
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.pose_landmarks,

            # POSE_CONNECTIONS tells MediaPipe which landmarks to connect with lines
            # e.g., it knows to draw a line from shoulder to elbow to wrist
            connections=self.mp_pose.POSE_CONNECTIONS,

            # Pre-made styles: landmark dots are colored by body part
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),

            # Connection lines are drawn in a subtle gray-white color
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(200, 200, 200),  # Light gray lines (BGR format)
                thickness=2,
                circle_radius=2
            )
        )

    def get_pixel_coordinates(self, landmark, frame_width, frame_height):
        """
        Converts a landmark's normalized coordinates (0.0–1.0) to actual pixel
        coordinates based on the frame's resolution.

        MediaPipe gives you normalized coordinates, but sometimes you need
        pixel positions — for example, to draw custom labels or measure distances.

        Args:
            landmark: A single landmark object (e.g., landmarks[LM.LEFT_KNEE]).
            frame_width (int): The width of the frame in pixels (e.g., 1280).
            frame_height (int): The height of the frame in pixels (e.g., 720).

        Returns:
            tuple: (x_pixels, y_pixels) as integers.

        Example:
            knee = landmarks[LM.LEFT_KNEE]
            kx, ky = detector.get_pixel_coordinates(knee, 1280, 720)
            cv2.circle(frame, (kx, ky), 10, (0, 255, 0), -1)  # Draw green dot
        """
        x_pixels = int(landmark.x * frame_width)
        y_pixels = int(landmark.y * frame_height)
        return (x_pixels, y_pixels)

    def is_landmark_visible(self, landmark, threshold=0.5):
        """
        Checks whether a specific landmark is reliably visible in the frame.

        MediaPipe gives each landmark a "visibility" score from 0.0 to 1.0.
        If a body part is behind the person, out of frame, or occluded,
        visibility will be low. Use this to skip unreliable data.

        Args:
            landmark: A single landmark object (e.g., landmarks[LM.LEFT_HIP]).
            threshold (float): Minimum visibility score to consider "visible".
                               0.5 is a reasonable default.

        Returns:
            bool: True if the landmark is visible enough to use, False otherwise.

        Example:
            if detector.is_landmark_visible(landmarks[LM.LEFT_KNEE]):
                # Safe to use this landmark
                angle = calculate_angle(...)
        """
        return landmark.visibility >= threshold

    def close(self):
        """
        Releases the MediaPipe Pose resources when you're done.
        Call this when the app is shutting down to free memory properly.

        Example:
            detector.close()
        """
        self.pose.close()
        print("[PoseDetector] Resources released.")


# ==============================================================================
# STANDALONE TEST
# Run this file directly to verify your webcam and pose detection work:
#     python utils/pose_detector.py
#
# You should see a window with your webcam feed and a skeleton drawn over you.
# Press 'Q' to quit.
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PoseDetector Standalone Test")
    print("  Stand in front of your webcam and move around!")
    print("  Press Q to quit.")
    print("=" * 60)

    # --- Initialize the detector ---
    detector = PoseDetector(
        detection_confidence=0.5,
        tracking_confidence=0.5
    )

    # --- Open the webcam ---
    # 0 = default webcam. Change to 1 or 2 if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure it's connected and not in use.")
        exit(1)

    # Set a reasonable resolution for testing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[TEST] Webcam opened. Starting detection loop...")

    # --- Main test loop ---
    while True:
        success, frame = cap.read()

        if not success:
            print("[WARNING] Couldn't read frame. Retrying...")
            continue

        # Flip horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # Run pose detection (skeleton will be drawn on frame automatically)
        landmarks = detector.detect(frame, draw_skeleton=True)

        # --- Display landmark info on screen ---
        frame_h, frame_w, _ = frame.shape  # Get frame dimensions

        if landmarks is not None:
            # Show a "DETECTED" status in green
            cv2.putText(frame, "Pose Detected!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)

            # Show some example landmark coordinates as text on screen
            # This helps you verify that the data looks reasonable
            debug_landmarks = {
                "L.Shoulder": LM.LEFT_SHOULDER,
                "L.Hip":      LM.LEFT_HIP,
                "L.Knee":     LM.LEFT_KNEE,
                "L.Ankle":    LM.LEFT_ANKLE,
            }

            y_offset = 100  # Starting Y position for text
            for name, index in debug_landmarks.items():
                lm = landmarks[index]
                # Convert to pixel coords for display
                px, py = detector.get_pixel_coordinates(lm, frame_w, frame_h)
                visible = "✓" if detector.is_landmark_visible(lm) else "✗"
                text = f"{name}: ({px}, {py})  vis={lm.visibility:.2f} {visible}"
                cv2.putText(frame, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 35

        else:
            # No pose found — show a warning in red
            cv2.putText(frame, "No pose detected — step into view!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # --- Show usage instructions ---
        cv2.putText(frame, "Press Q to quit", (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # --- Display the frame in a window ---
        cv2.imshow("PoseDetector Test — utils/pose_detector.py", frame)

        # --- Check for quit key ---
        # cv2.waitKey(1) waits 1ms for a keypress. & 0xFF normalizes the key code.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[TEST] Q pressed — quitting.")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[TEST] Test complete. If you saw your skeleton, everything is working!")