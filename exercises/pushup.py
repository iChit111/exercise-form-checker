"""
exercises/pushup.py — Push-Up Form Analyzer
=============================================
YOUR ROLE: Physics & Exercise Logic (Person 3)

What this file does:
    Analyzes a push-up using body landmark positions from MediaPipe.
    It checks joint angles for elbow and body alignment, counts reps,
    and returns a form status.

Push-Up Biomechanics (the physics behind it):
----------------------------------------------
A proper push-up involves these joint angles:

1. ELBOW ANGLE (Shoulder → Elbow → Wrist):
   - Arms fully extended (top of push-up):  ~160–180°
   - Good depth (bottom of push-up):         ~70–90°
   - Not going low enough:                   >90° at bottom
   - Collapsing elbows too much:             <60°

2. BODY ALIGNMENT — Hip-Shoulder-Ankle should form a straight line:
   - The body should be a flat "plank" from head to heels
   - If the hips sag DOWN: hip y-coordinate drops below the shoulder-ankle line
   - If the hips pike UP:  hip y-coordinate rises above the shoulder-ankle line
   - We check this by comparing the hip's vertical position to the shoulder and ankle

Rep Counting Logic:
-------------------
    State "UP"   = arms extended (elbow angle > 150°)
    State "DOWN" = arms bent at bottom (elbow angle < 90°)
    DOWN → UP transition = one completed rep
"""

from utils.angle_calculator import calculate_angle, get_landmark_coords

# ==============================================================================
# LANDMARK INDEX CONSTANTS
# ==============================================================================
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

# ==============================================================================
# ANGLE THRESHOLDS
# ==============================================================================
ELBOW_ANGLE_EXTENDED    = 150   # Above this = arms extended (top position)
ELBOW_ANGLE_GOOD_DEPTH  = 90    # Below this = good push-up depth
ELBOW_ANGLE_TOO_LOW     = 60    # Below this = elbow collapsing too much
HIP_SAG_TOLERANCE       = 0.05  # How much hip can deviate from the shoulder-ankle line (normalized units)

# ==============================================================================
# REP COUNTER STATE
# ==============================================================================
_pushup_state = {
    "phase": "UP",
    "reps": 0
}


def analyze_pushup(landmarks):
    """
    Analyzes a single frame of a push-up using body landmarks.

    Args:
        landmarks: A list of 33 MediaPipe NormalizedLandmark objects.

    Returns:
        dict: {
            "angles": {
                "left_elbow":  float,
                "right_elbow": float,
            },
            "reps":        int,
            "form_status": str   # "good", "not_deep_enough", "too_low", "hips_sagging", "hips_piking"
        }
        Returns None if landmarks are missing or unreliable.
    """

    if landmarks is None:
        return None

    try:
        left_shoulder  = get_landmark_coords(landmarks[LEFT_SHOULDER])
        right_shoulder = get_landmark_coords(landmarks[RIGHT_SHOULDER])
        left_elbow     = get_landmark_coords(landmarks[LEFT_ELBOW])
        right_elbow    = get_landmark_coords(landmarks[RIGHT_ELBOW])
        left_wrist     = get_landmark_coords(landmarks[LEFT_WRIST])
        right_wrist    = get_landmark_coords(landmarks[RIGHT_WRIST])
        left_hip       = get_landmark_coords(landmarks[LEFT_HIP])
        right_hip      = get_landmark_coords(landmarks[RIGHT_HIP])
        left_ankle     = get_landmark_coords(landmarks[LEFT_ANKLE])
        right_ankle    = get_landmark_coords(landmarks[RIGHT_ANKLE])
    except (IndexError, AttributeError):
        return None

    # Check visibility
    critical = [
        landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER],
        landmarks[LEFT_ELBOW],    landmarks[RIGHT_ELBOW],
        landmarks[LEFT_WRIST],    landmarks[RIGHT_WRIST],
    ]
    if any(lm.visibility < 0.5 for lm in critical):
        return None

    # ==============================================================================
    # CALCULATE JOINT ANGLES
    # ==============================================================================

    # ELBOW ANGLE: Shoulder → Elbow → Wrist
    # Fully extended = ~180°, bent at bottom of push-up = ~70–90°
    left_elbow_angle  = calculate_angle(left_shoulder,  left_elbow,  left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    avg_elbow_angle   = (left_elbow_angle + right_elbow_angle) / 2

    # ==============================================================================
    # BODY ALIGNMENT CHECK — is the body forming a straight plank?
    # We check this by seeing if the HIP Y-position is between the SHOULDER
    # and ANKLE Y-positions (in normalized coordinates, Y increases downward).
    #
    # In a side-on push-up:
    #   Shoulder Y and Ankle Y should be roughly equal (same height)
    #   Hip Y should be very close to that same line
    #
    # If Hip Y is MUCH LOWER than the shoulder-ankle midpoint → hips are sagging
    # If Hip Y is MUCH HIGHER than the shoulder-ankle midpoint → hips are piking
    # ==============================================================================
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y      = (left_hip[1]      + right_hip[1])      / 2
    avg_ankle_y    = (left_ankle[1]    + right_ankle[1])    / 2

    # Ideal hip Y = average of shoulder Y and ankle Y (midpoint of the plank line)
    ideal_hip_y    = (avg_shoulder_y + avg_ankle_y) / 2
    hip_deviation  = avg_hip_y - ideal_hip_y  # Positive = hips lower, Negative = hips higher

    # ==============================================================================
    # REP COUNTING
    # ==============================================================================
    if avg_elbow_angle < ELBOW_ANGLE_GOOD_DEPTH:
        _pushup_state["phase"] = "DOWN"

    if avg_elbow_angle > ELBOW_ANGLE_EXTENDED and _pushup_state["phase"] == "DOWN":
        _pushup_state["phase"] = "UP"
        _pushup_state["reps"] += 1

    # ==============================================================================
    # FORM STATUS
    # ==============================================================================
    form_status = "good"

    # Check body alignment FIRST — alignment issues are the most important
    if hip_deviation > HIP_SAG_TOLERANCE:
        form_status = "hips_sagging"
    elif hip_deviation < -HIP_SAG_TOLERANCE:
        form_status = "hips_piking"
    elif avg_elbow_angle > ELBOW_ANGLE_GOOD_DEPTH and _pushup_state["phase"] == "DOWN":
        # They're supposed to be going down but haven't reached good depth
        form_status = "not_deep_enough"
    elif avg_elbow_angle < ELBOW_ANGLE_TOO_LOW:
        form_status = "too_low"

    return {
        "angles": {
            "left_elbow":  left_elbow_angle,
            "right_elbow": right_elbow_angle,
        },
        "reps":        _pushup_state["reps"],
        "form_status": form_status
    }


def reset_pushup_counter():
    """Resets the push-up rep counter back to zero."""
    _pushup_state["phase"] = "UP"
    _pushup_state["reps"]  = 0
    print("[Push-Up] Rep counter reset.")