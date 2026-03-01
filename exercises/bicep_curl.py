"""
exercises/bicep_curl.py — Bicep Curl Form Analyzer
====================================================
YOUR ROLE: Physics & Exercise Logic (Person 3)

What this file does:
    Analyzes a bicep curl using body landmark positions from MediaPipe.
    Checks the elbow angle, detects swinging (cheating), counts reps,
    and returns a form status.

Bicep Curl Biomechanics (the physics behind it):
-------------------------------------------------
The bicep curl is a single-joint movement focused on the elbow:

1. ELBOW ANGLE (Shoulder → Elbow → Wrist):
   - Starting (arm extended):    ~150–170°
   - Good curl (top position):   ~30–50°
   - Not curling enough:         >60° at top
   - Over-curling (rare):        <20°

2. SHOULDER STABILITY (detecting "swinging/cheating"):
   - During a strict curl, the UPPER ARM should stay still.
   - Cheating = the shoulder moves forward/up to help lift the weight.
   - We detect this by tracking the SHOULDER Y-coordinate over time.
   - If the shoulder moves significantly UP during the curl, the user is swinging.

Rep Counting Logic:
-------------------
    State "DOWN" = arm extended (elbow angle > 150°)
    State "UP"   = arm curled (elbow angle < 50°)
    DOWN → UP → DOWN cycle = one completed rep
    (We count when going from UP back to DOWN, i.e., the full curl is complete)
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

# ==============================================================================
# ANGLE THRESHOLDS
# ==============================================================================
ELBOW_ANGLE_EXTENDED      = 150   # Above this = arm is extended (start position)
ELBOW_ANGLE_CURLED        = 50    # Below this = arm is fully curled (top position)
ELBOW_ANGLE_NOT_ENOUGH    = 60    # Above this at top = not curling enough
SHOULDER_SWING_TOLERANCE  = 0.04  # How much shoulder Y can move before flagging swing (normalized units)

# ==============================================================================
# REP COUNTER STATE
# We also track the shoulder's Y position to detect swinging.
# ==============================================================================
_curl_state = {
    "phase": "DOWN",          # "DOWN" = arm extended, "UP" = arm curled
    "reps": 0,
    "shoulder_y_baseline": None  # Shoulder Y at the start of each rep (to detect swinging)
}


def analyze_bicep_curl(landmarks):
    """
    Analyzes a single frame of a bicep curl using body landmarks.

    Args:
        landmarks: A list of 33 MediaPipe NormalizedLandmark objects.

    Returns:
        dict: {
            "angles": {
                "left_elbow":  float,
                "right_elbow": float,
            },
            "reps":        int,
            "form_status": str   # "good", "not_curled_enough", "swinging"
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
    except (IndexError, AttributeError):
        return None

    # Check visibility of arm landmarks
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
    # Extended arm = ~160°, fully curled = ~30–50°
    left_elbow_angle  = calculate_angle(left_shoulder,  left_elbow,  left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Use the average of both arms for the rep counter
    # (Assumes user is doing both arms simultaneously; adapt if doing one arm at a time)
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

    # ==============================================================================
    # SHOULDER SWING DETECTION
    # Track how much the shoulder moves vertically during the curl.
    # In MediaPipe, Y increases DOWNWARD, so a shoulder moving UP = Y decreasing.
    # If Y decreases significantly = user is swinging their body to help lift.
    # ==============================================================================
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # Set the baseline shoulder position when the arm is extended (start of rep)
    if avg_elbow_angle > ELBOW_ANGLE_EXTENDED:
        _curl_state["shoulder_y_baseline"] = avg_shoulder_y

    # Detect swing: shoulder moved UP more than the tolerance during the curl
    is_swinging = False
    if _curl_state["shoulder_y_baseline"] is not None:
        shoulder_movement = _curl_state["shoulder_y_baseline"] - avg_shoulder_y
        # Positive value = shoulder moved UP (Y decreased) = possible swinging
        if shoulder_movement > SHOULDER_SWING_TOLERANCE:
            is_swinging = True

    # ==============================================================================
    # REP COUNTING
    # ==============================================================================
    # When the arm is fully curled, mark phase as "UP"
    if avg_elbow_angle < ELBOW_ANGLE_CURLED:
        _curl_state["phase"] = "UP"

    # When the arm returns to extended after being curled = one complete rep
    if avg_elbow_angle > ELBOW_ANGLE_EXTENDED and _curl_state["phase"] == "UP":
        _curl_state["phase"] = "DOWN"
        _curl_state["reps"] += 1

    # ==============================================================================
    # FORM STATUS
    # ==============================================================================
    form_status = "good"

    if is_swinging:
        form_status = "swinging"
    elif _curl_state["phase"] == "UP" and avg_elbow_angle > ELBOW_ANGLE_NOT_ENOUGH:
        # They're in the "up" phase but haven't curled high enough
        form_status = "not_curled_enough"

    return {
        "angles": {
            "left_elbow":  left_elbow_angle,
            "right_elbow": right_elbow_angle,
        },
        "reps":        _curl_state["reps"],
        "form_status": form_status
    }


def reset_curl_counter():
    """Resets the bicep curl rep counter back to zero."""
    _curl_state["phase"]             = "DOWN"
    _curl_state["reps"]              = 0
    _curl_state["shoulder_y_baseline"] = None
    print("[Bicep Curl] Rep counter reset.")