"""
utils/angle_calculator.py — Joint Angle Calculator
====================================================
YOUR ROLE: Physics & Math (Person 3)
This file provides the core mathematical tool used by ALL exercise modules.

What this file does:
    Takes three body points (A, B, C) and calculates the angle at point B.
    This is the foundation of form checking — almost every exercise check
    boils down to "what angle is this joint at?"

The Physics & Math Behind It:
------------------------------
Imagine your leg during a squat. You have three points:
    A = Hip
    B = Knee   ← the joint we care about
    C = Ankle

To find the angle at the KNEE, we use VECTORS and the DOT PRODUCT formula.

Step 1 — Create two vectors pointing AWAY from B:
    Vector BA = A - B   (from knee toward hip)
    Vector BC = C - B   (from knee toward ankle)

Step 2 — Use the dot product formula:
    cos(θ) = (BA · BC) / (|BA| × |BC|)

    Where:
        BA · BC  = dot product = (BA.x × BC.x) + (BA.y × BC.y)
        |BA|     = magnitude (length) of vector BA = √(BA.x² + BA.y²)
        |BA|×|BC|= product of both magnitudes

Step 3 — Solve for θ (theta):
    θ = arccos(cos(θ))   → gives us the angle in RADIANS
    θ_degrees = θ × (180 / π)   → convert to degrees (easier to understand)

Why degrees?
    A straight leg = ~180°
    A fully bent knee = ~90° or less
    This matches how biomechanics textbooks describe joint angles,
    making it easy to define "good form" thresholds.
"""

import numpy as np  # NumPy handles all the math efficiently


def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at joint B, formed by points A-B-C.

    This is the PRIMARY function used by all exercise analysis files.
    Think of it like a protractor placed at point B, measuring the opening
    between the lines B→A and B→C.

    Args:
        a (tuple or array-like): Coordinates of the FIRST point (e.g., hip).
                                 Format: (x, y) where x and y are floats.
        b (tuple or array-like): Coordinates of the JOINT/VERTEX point (e.g., knee).
                                 The angle is measured HERE.
        c (tuple or array-like): Coordinates of the THIRD point (e.g., ankle).
                                 Format: (x, y) where x and y are floats.

    Returns:
        float: The angle at point B in DEGREES (0° to 180°).
               Returns 0.0 if the calculation fails (e.g., points are identical).

    Example usage (in an exercise file):
        # Get the knee angle during a squat
        hip   = [landmarks[LM.LEFT_HIP].x,   landmarks[LM.LEFT_HIP].y]
        knee  = [landmarks[LM.LEFT_KNEE].x,  landmarks[LM.LEFT_KNEE].y]
        ankle = [landmarks[LM.LEFT_ANKLE].x, landmarks[LM.LEFT_ANKLE].y]
        angle = calculate_angle(hip, knee, ankle)
        # angle ≈ 90° means the knee is bent to a right angle (good squat depth)
        # angle ≈ 170° means the knee is nearly straight (standing position)
    """

    # --- Convert inputs to NumPy arrays ---
    # This allows us to do math on them easily (subtraction, dot product, etc.)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # --- Step 1: Create two vectors pointing AWAY from point B ---
    # Think of these as two "arms" extending from the joint
    vector_ba = a - b   # Direction from B toward A (e.g., knee → hip)
    vector_bc = c - b   # Direction from B toward C (e.g., knee → ankle)

    # --- Step 2: Calculate the magnitudes (lengths) of each vector ---
    # Magnitude = the straight-line distance from B to A (or B to C)
    # np.linalg.norm() computes √(x² + y²) for us
    magnitude_ba = np.linalg.norm(vector_ba)
    magnitude_bc = np.linalg.norm(vector_bc)

    # --- Safety check: Avoid division by zero ---
    # This can happen if two points are at the exact same position
    # (e.g., landmark detection glitch). We return 0.0 as a safe fallback.
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    # --- Step 3: Calculate the dot product ---
    # Dot product = BA.x × BC.x + BA.y × BC.y
    # This gives us a number related to how "aligned" the two vectors are.
    dot_product = np.dot(vector_ba, vector_bc)

    # --- Step 4: Apply the dot product formula to find cos(θ) ---
    # cos(θ) = dot_product / (|BA| × |BC|)
    # np.clip keeps the value between -1.0 and 1.0 to prevent floating-point
    # errors from causing arccos to fail (arccos only works on values in [-1, 1])
    cos_angle = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)

    # --- Step 5: Find the angle using arccos, then convert to degrees ---
    # np.arccos() returns the angle in RADIANS
    # np.degrees() converts radians → degrees (multiply by 180/π)
    angle_degrees = np.degrees(np.arccos(cos_angle))

    return round(angle_degrees, 2)  # Round to 2 decimal places for clean output


def get_landmark_coords(landmark):
    """
    Helper function: Extracts (x, y) coordinates from a MediaPipe landmark object.

    MediaPipe landmarks have .x and .y attributes (normalized 0.0–1.0).
    This function pulls them into a simple list that calculate_angle() can use.

    Args:
        landmark: A single MediaPipe NormalizedLandmark object.
                  (e.g., landmarks[LM.LEFT_KNEE])

    Returns:
        list: [x, y] as floats.

    Example:
        knee_coords = get_landmark_coords(landmarks[LM.LEFT_KNEE])
        # knee_coords = [0.52, 0.73]  ← normalized screen position
    """
    return [landmark.x, landmark.y]


# ==============================================================================
# QUICK SELF-TEST
# Run this file directly to verify the math is working:
#     python utils/angle_calculator.py
# ==============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  angle_calculator.py — Self Test")
    print("=" * 50)

    # Test 1: Right angle (90°)
    # Imagine a knee bent at exactly 90 degrees
    # A=hip is directly above B=knee, C=ankle is directly to the right
    a = [0, 0]   # hip (directly above knee)
    b = [0, 1]   # knee (the joint)
    c = [1, 1]   # ankle (directly to the right of knee)
    angle = calculate_angle(a, b, c)
    print(f"\nTest 1 — Right angle (expected ~90°): {angle}°")
    assert abs(angle - 90.0) < 0.01, "Test 1 FAILED!"
    print("  ✓ PASSED")

    # Test 2: Straight line (180°)
    # Imagine a fully extended leg — hip, knee, and ankle in a straight line
    a = [0, 0]   # hip
    b = [0, 1]   # knee
    c = [0, 2]   # ankle (all in a straight vertical line)
    angle = calculate_angle(a, b, c)
    print(f"\nTest 2 — Straight line (expected ~180°): {angle}°")
    assert abs(angle - 180.0) < 0.01, "Test 2 FAILED!"
    print("  ✓ PASSED")

    # Test 3: 45-degree angle
    # B is at origin. Vector BA points RIGHT, vector BC points diagonally (45° from BA).
    # The angle BETWEEN these two vectors at point B = 45°.
    a = [1, 0]   # A is to the RIGHT of B → vector BA points right
    b = [0, 0]   # B is the joint (vertex)
    c = [1, 1]   # C is diagonally up-right → vector BC is 45° away from BA
    angle = calculate_angle(a, b, c)
    print(f"\nTest 3 — ~45° angle (expected ~45°): {angle}°")
    assert abs(angle - 45.0) < 1.0, "Test 3 FAILED!"
    print("  ✓ PASSED")

    print("\n" + "=" * 50)
    print("  All tests passed! angle_calculator.py is ready.")
    print("=" * 50)