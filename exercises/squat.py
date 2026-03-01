"""
exercises/squat.py — Squat Form Analyzer
=========================================
YOUR ROLE: Physics & Exercise Logic (Person 3)

What this file does:
    Analyzes a squat using body landmark positions from MediaPipe.
    It checks joint angles, counts reps, and returns a form status.

Squat Biomechanics (the physics behind it):
--------------------------------------------
A proper squat involves these joint angles:

1. KNEE ANGLE (Hip → Knee → Ankle):
   - Standing position:  ~170–180° (nearly straight)
   - Good squat depth:   ~90–110°  (thighs parallel to floor)
   - Too shallow:        >110°     (not enough depth)
   - Too deep/dangerous: <70°      (extreme flexion, stress on joints)

2. HIP ANGLE (Shoulder → Hip → Knee):
   - Standing position:  ~170–180°
   - Good squat:         ~80–100°  (hips hinge back, trunk leans slightly)
   - Too upright:        >120°     (insufficient hip hinge)

3. BACK ALIGNMENT (visual check via shoulder-hip-knee vertical):
   - The spine should stay roughly neutral (not rounding or hyperextending)
   - We approximate this by checking if hips stay between shoulders and ankles

Rep Counting Logic:
-------------------
We use a simple STATE MACHINE:
    State "UP"   = user is standing (knee angle > 160°)
    State "DOWN" = user is in squat (knee angle < 110°)

When the user goes from DOWN → UP, we count ONE rep.
This prevents counting partial movements or jitter as reps.
"""

from utils.angle_calculator import calculate_angle, get_landmark_coords

# ==============================================================================
# LANDMARK INDEX CONSTANTS (from pose_detector.py — Person 2's LM class)
# We redefine them here as plain integers so this file works independently.
# These numbers match MediaPipe's official landmark indices.
# ==============================================================================
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

# ==============================================================================
# ANGLE THRESHOLDS — the "rules" that define good vs. bad squat form
# Changing these numbers adjusts how strict the form checker is.
# ==============================================================================
KNEE_ANGLE_STANDING   = 160   # Above this = user is standing up (degrees)
KNEE_ANGLE_GOOD_DEPTH = 110   # Below this = good squat depth reached (degrees)
KNEE_ANGLE_TOO_DEEP   = 70    # Below this = potentially too deep (degrees)
HIP_ANGLE_GOOD        = 120   # Below this = good hip hinge (degrees)

# ==============================================================================
# REP COUNTER STATE — stored as a mutable dict so it persists between calls.
# (In Python, a module-level dict is shared across calls to analyze_squat())
# ==============================================================================
_squat_state = {
    "phase": "UP",    # Current phase: "UP" (standing) or "DOWN" (squatting)
    "reps": 0         # Total completed reps
}


def analyze_squat(landmarks):
    """
    Analyzes a single frame of a squat using body landmarks.

    This function is called EVERY FRAME by main.py (via run_app).
    It checks joint angles, updates rep count, and returns a status dictionary.

    Args:
        landmarks: A list of 33 MediaPipe NormalizedLandmark objects.
                   Comes from PoseDetector.detect(frame) in pose_detector.py.
                   Each landmark has .x, .y (normalized 0.0–1.0) and .visibility.

    Returns:
        dict: A dictionary with exactly these keys (main.py depends on this format):
            {
                "angles": {
                    "left_knee":  float,   # Knee angle in degrees
                    "right_knee": float,
                    "left_hip":   float,   # Hip angle in degrees
                    "right_hip":  float,
                },
                "reps":        int,        # Total completed reps
                "form_status": str         # "good", "too_shallow", "too_deep", "adjust_hips"
            }
        Returns None if landmarks are missing or unreliable.
    """

    # --- Safety check: Make sure landmarks exist ---
    if landmarks is None:
        return None

    # --- Extract coordinates for all relevant joints ---
    # Each call to get_landmark_coords() returns [x, y] for that landmark
    try:
        left_shoulder  = get_landmark_coords(landmarks[LEFT_SHOULDER])
        right_shoulder = get_landmark_coords(landmarks[RIGHT_SHOULDER])
        left_hip       = get_landmark_coords(landmarks[LEFT_HIP])
        right_hip      = get_landmark_coords(landmarks[RIGHT_HIP])
        left_knee      = get_landmark_coords(landmarks[LEFT_KNEE])
        right_knee     = get_landmark_coords(landmarks[RIGHT_KNEE])
        left_ankle     = get_landmark_coords(landmarks[LEFT_ANKLE])
        right_ankle    = get_landmark_coords(landmarks[RIGHT_ANKLE])
    except (IndexError, AttributeError):
        # Landmarks list is incomplete or malformed
        return None

    # --- Check visibility of critical landmarks ---
    # If key joints aren't visible, the analysis won't be reliable
    critical = [
        landmarks[LEFT_HIP], landmarks[RIGHT_HIP],
        landmarks[LEFT_KNEE], landmarks[RIGHT_KNEE],
        landmarks[LEFT_ANKLE], landmarks[RIGHT_ANKLE]
    ]
    if any(lm.visibility < 0.5 for lm in critical):
        return None  # Too many joints hidden — skip this frame

    # ==============================================================================
    # CALCULATE JOINT ANGLES
    # Using the vector dot product method from angle_calculator.py
    # ==============================================================================

    # KNEE ANGLE: Hip → Knee → Ankle
    # A straight leg = ~180°, a bent knee = ~90°
    left_knee_angle  = calculate_angle(left_hip,  left_knee,  left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # HIP ANGLE: Shoulder → Hip → Knee
    # This measures how much the torso is leaning forward relative to the thigh
    left_hip_angle  = calculate_angle(left_shoulder,  left_hip,  left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # Use the AVERAGE of left and right sides for a more stable reading
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    avg_hip_angle  = (left_hip_angle  + right_hip_angle)  / 2

    # ==============================================================================
    # REP COUNTING — State Machine Logic
    # ==============================================================================
    # Check if user is in the DOWN (squat) position
    if avg_knee_angle < KNEE_ANGLE_GOOD_DEPTH:
        _squat_state["phase"] = "DOWN"

    # Check if user has returned to UP (standing) — this completes a rep
    if avg_knee_angle > KNEE_ANGLE_STANDING and _squat_state["phase"] == "DOWN":
        _squat_state["phase"] = "UP"
        _squat_state["reps"] += 1

    # ==============================================================================
    # FORM STATUS — Determine feedback category based on angles
    # ==============================================================================
    form_status = "good"  # Default: assume good form

    if avg_knee_angle > KNEE_ANGLE_GOOD_DEPTH:
        # Knee angle is too large — they haven't squatted deep enough
        form_status = "too_shallow"

    elif avg_knee_angle < KNEE_ANGLE_TOO_DEEP:
        # Knee angle is very small — they may be squatting too deep
        form_status = "too_deep"

    elif avg_hip_angle > HIP_ANGLE_GOOD:
        # Hip isn't hinging enough — torso is too upright or knees are too forward
        form_status = "adjust_hips"

    # ==============================================================================
    # RETURN THE RESULT DICTIONARY
    # main.py passes this to get_feedback() and update_rep_count()
    # ==============================================================================
    return {
        "angles": {
            "left_knee":  left_knee_angle,
            "right_knee": right_knee_angle,
            "left_hip":   left_hip_angle,
            "right_hip":  right_hip_angle,
        },
        "reps":        _squat_state["reps"],
        "form_status": form_status
    }


def reset_squat_counter():
    """
    Resets the squat rep counter back to zero.
    Call this when the user switches exercises or clicks a reset button in the UI.
    """
    _squat_state["phase"] = "UP"
    _squat_state["reps"]  = 0
    print("[Squat] Rep counter reset.")