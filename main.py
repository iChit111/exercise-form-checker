"""
main.py — Exercise Form Checker
================================
This is the entry point for the entire application.
When someone runs `python main.py`, this file starts everything up.

Project Structure Reminder:
    main.py                  ← YOU ARE HERE (Project Lead)
    exercises/
        squat.py             ← Squat analysis logic (teammate)
        pushup.py            ← Push-up analysis logic (teammate)
        bicep_curl.py        ← Bicep curl analysis logic (teammate)
    utils/
        pose_detector.py     ← MediaPipe pose detection wrapper (teammate)
        angle_calculator.py  ← NumPy angle math helpers (teammate)
        feedback.py          ← Feedback/coaching message logic (teammate)
    ui/
        app_window.py        ← PyQt5 desktop window & UI (teammate)
    assets/                  ← Images, icons, sounds, etc.
"""

import sys  # Needed to pass command-line arguments to the PyQt5 app and to exit cleanly


# ==============================================================================
# SECTION 1: THIRD-PARTY LIBRARY IMPORTS
# These are packages installed via requirements.txt.
# If you get an ImportError, run: pip install -r requirements.txt
# ==============================================================================

import cv2          # OpenCV — handles webcam feed and video frame processing
import numpy as np  # NumPy — used for math operations like angle calculations

from PyQt5.QtWidgets import QApplication  # The core PyQt5 class that manages the desktop app


# ==============================================================================
# SECTION 2: LOCAL MODULE IMPORTS (written by your teammates)
# These will cause errors until your teammates create their files.
# That's okay! Comment them out temporarily if you need to test main.py alone.
# ==============================================================================

# --- UI Module ---
# Teammate responsible: [assign someone]
# Expected: A class called AppWindow that creates the main desktop window.
# The window should display the webcam feed, exercise selector, and feedback panel.
from ui.app_window import AppWindow

# --- Utility Modules ---
# Teammate responsible: [assign someone]
# Expected: A class called PoseDetector with a method like .detect(frame)
# that returns pose landmark data from a webcam frame using MediaPipe.
from utils.pose_detector import PoseDetector

# Expected: A function or class that takes landmark positions and returns
# joint angles (e.g., knee angle, elbow angle) as numbers in degrees.
from utils.angle_calculator import calculate_angle

# Expected: A function or class that takes an angle (or angles) and the
# current exercise, and returns a feedback string like "Go deeper!" or "Good form!".
from utils.feedback import get_feedback

# --- Exercise Modules ---
# Each file should contain the logic specific to that exercise.
# Teammate responsible: [assign someone per exercise]
# Expected: Each module should expose a function or class that accepts pose
# landmarks and returns structured data (angles, rep count, form status, etc.)
from exercises.squat import analyze_squat
from exercises.pushup import analyze_pushup
from exercises.bicep_curl import analyze_bicep_curl


# ==============================================================================
# SECTION 3: CONSTANTS & CONFIGURATION
# Central place to change app-wide settings. No magic numbers scattered around!
# ==============================================================================

APP_TITLE = "Exercise Form Checker"  # Shown in the window title bar
WEBCAM_INDEX = 0                     # 0 = default webcam. Try 1 or 2 if yours doesn't work.
FRAME_WIDTH = 1280                   # Webcam capture width in pixels
FRAME_HEIGHT = 720                   # Webcam capture height in pixels

# The list of exercises the user can choose from in the UI.
# Keys must match what the exercise modules and UI dropdown expect.
SUPPORTED_EXERCISES = {
    "Squat": analyze_squat,
    "Push-Up": analyze_pushup,
    "Bicep Curl": analyze_bicep_curl,
}


# ==============================================================================
# SECTION 4: WEBCAM SETUP
# This function initializes the webcam using OpenCV.
# It's separated into its own function to keep things clean and easy to debug.
# ==============================================================================

def initialize_webcam(index=WEBCAM_INDEX):
    """
    Opens a connection to the webcam.

    Args:
        index (int): The webcam device index. Default is 0 (built-in webcam).

    Returns:
        cv2.VideoCapture: The webcam capture object, or None if it failed.
    """
    print(f"[SETUP] Connecting to webcam (device index: {index})...")
    cap = cv2.VideoCapture(index)

    # Set the resolution we want for the webcam stream
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that it's plugged in and not in use.")
        return None

    print("[SETUP] Webcam connected successfully.")
    return cap


# ==============================================================================
# SECTION 5: CORE APPLICATION LOGIC
# This function ties everything together — it's the "brain" of the app.
# It reads webcam frames, runs pose detection, analyzes form, and sends
# the results to the UI.
# ==============================================================================

def run_app(app_window, pose_detector, webcam):
    """
    The main processing loop. This is called once per frame while the app runs.

    It does the following steps every frame:
        1. Read a frame from the webcam
        2. Detect the user's pose landmarks using MediaPipe
        3. Determine which exercise is currently selected in the UI
        4. Run that exercise's analysis function
        5. Get coaching feedback based on the analysis
        6. Send the annotated frame and feedback back to the UI to display

    Args:
        app_window (AppWindow): The main UI window instance.
        pose_detector (PoseDetector): The pose detection utility instance.
        webcam (cv2.VideoCapture): The active webcam capture object.
    """

    # --- Step 1: Read a frame from the webcam ---
    # cap.read() returns two things:
    #   success (bool): Was the frame read successfully?
    #   frame (numpy array): The actual image data (rows x cols x 3 color channels)
    success, frame = webcam.read()

    if not success:
        print("[WARNING] Failed to read frame from webcam. Skipping...")
        return  # Skip this frame and try again next time

    # Flip the frame horizontally so it acts like a mirror — feels more natural for the user
    frame = cv2.flip(frame, 1)

    # --- Step 2: Detect pose landmarks ---
    # PLACEHOLDER: Once pose_detector.py is ready, this will return landmark data.
    # For now, landmarks will be None and the exercise analysis will be skipped.
    #
    # Expected output: A MediaPipe Pose result object, or a simplified dict of
    # landmark coordinates — coordinate this format with the pose_detector teammate.
    landmarks = pose_detector.detect(frame)  # ← teammate fills this in

    # --- Step 3: Find out which exercise the user has selected in the UI ---
    # PLACEHOLDER: AppWindow should have a method like .get_selected_exercise()
    # that returns one of the keys from SUPPORTED_EXERCISES (e.g., "Squat").
    selected_exercise_name = app_window.get_selected_exercise()  # ← teammate fills this in

    # --- Step 4: Run the appropriate exercise analysis ---
    analysis_result = None  # Will hold angle data, rep count, form status, etc.

    if landmarks is not None and selected_exercise_name in SUPPORTED_EXERCISES:
        # Look up the right analysis function based on the user's selection
        analyze_function = SUPPORTED_EXERCISES[selected_exercise_name]

        # PLACEHOLDER: Each analyze_*() function should accept landmarks and return
        # a dictionary like: { "angles": {...}, "reps": int, "form_status": str }
        # Coordinate the exact format with each exercise module teammate.
        analysis_result = analyze_function(landmarks)  # ← teammate fills this in

    # --- Step 5: Generate coaching feedback ---
    feedback_message = "Waiting for pose detection..."  # Default message

    if analysis_result is not None:
        # PLACEHOLDER: get_feedback() should accept the analysis result and return
        # a human-readable string like "Knees caving in — push them outward!"
        feedback_message = get_feedback(analysis_result, selected_exercise_name)  # ← teammate fills this in

    # --- Step 6: Send results to the UI ---
    # PLACEHOLDER: AppWindow should have methods to:
    #   .update_video_feed(frame)    — display the current webcam frame
    #   .update_feedback(message)    — show the coaching feedback text
    #   .update_rep_count(count)     — update the rep counter display
    #
    # These are called every frame to keep the UI live and responsive.
    app_window.update_video_feed(frame)          # ← teammate fills this in
    app_window.update_feedback(feedback_message) # ← teammate fills this in

    if analysis_result is not None:
        rep_count = analysis_result.get("reps", 0)
        app_window.update_rep_count(rep_count)   # ← teammate fills this in


# ==============================================================================
# SECTION 6: CLEANUP
# Always release resources when the app closes. Failing to do this can leave
# the webcam "stuck" open even after the program ends.
# ==============================================================================

def cleanup(webcam):
    """
    Releases the webcam and closes any OpenCV windows.
    Call this when the app is shutting down.

    Args:
        webcam (cv2.VideoCapture): The webcam capture object to release.
    """
    print("[SHUTDOWN] Releasing webcam and cleaning up...")
    if webcam is not None:
        webcam.release()
    cv2.destroyAllWindows()
    print("[SHUTDOWN] Done. Goodbye!")


# ==============================================================================
# SECTION 7: ENTRY POINT
# This block only runs when you execute `python main.py` directly.
# It does NOT run if this file is imported by another module (which is good
# practice — it keeps things flexible and testable).
# ==============================================================================

if __name__ == "__main__":
    print(f"[STARTUP] Starting {APP_TITLE}...")

    # --- Initialize the PyQt5 application ---
    # Every PyQt5 app needs one QApplication instance. sys.argv passes any
    # command-line arguments (usually just the script name).
    qt_app = QApplication(sys.argv)

    # --- Set up the webcam ---
    webcam = initialize_webcam()
    if webcam is None:
        print("[ERROR] Could not start webcam. Exiting.")
        sys.exit(1)  # Exit with error code 1 so the OS knows something went wrong

    # --- Initialize the Pose Detector ---
    # PLACEHOLDER: PoseDetector.__init__() might accept config options like
    # detection confidence threshold. Coordinate with that teammate.
    pose_detector = PoseDetector()  # ← teammate fills this in

    # --- Create and show the main UI window ---
    # PLACEHOLDER: AppWindow might need arguments like the exercise list or app title.
    # Pass SUPPORTED_EXERCISES so the UI can populate its dropdown menu.
    app_window = AppWindow(
        title=APP_TITLE,
        exercises=list(SUPPORTED_EXERCISES.keys())
    )  # ← teammate fills this in
    app_window.show()

    # --- Connect the frame processing loop to the UI timer ---
    # PyQt5 apps are event-driven, meaning they wait for things to happen (clicks,
    # timer ticks, etc.) rather than running a traditional while loop.
    # We use a QTimer to call run_app() repeatedly — once per "tick" = one frame.
    #
    # PLACEHOLDER: AppWindow should set up a QTimer internally and expose a method
    # like .set_frame_callback(fn) that calls our run_app function each tick.
    # Alternatively, app_window.py can import and call run_app directly.
    app_window.set_frame_callback(
        lambda: run_app(app_window, pose_detector, webcam)
    )  # ← teammate fills this in

    print(f"[STARTUP] {APP_TITLE} is running. Close the window to exit.")

    # --- Start the PyQt5 event loop ---
    # qt_app.exec_() hands control to PyQt5. It blocks here until the window
    # is closed, then returns an exit code (0 = success, anything else = error).
    exit_code = qt_app.exec_()

    # --- Clean up before fully exiting ---
    cleanup(webcam)
    sys.exit(exit_code)