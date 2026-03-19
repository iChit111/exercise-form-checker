"""
utils/feedback.py — Coaching Feedback Message Generator
=========================================================
YOUR ROLE: Feedback System
This file is responsible for turning raw exercise analysis data into
friendly, human-readable coaching messages displayed in the UI.

How it fits into the project:
    main.py calls:  get_feedback(analysis_result, exercise_name)
    You return:     A string like "✅ Great squat depth!" or "⚠️ Go deeper!"

The analysis_result dict looks like this:
    {
        "angles":      { ... },      # Joint angles in degrees (from angle_calculator.py)
        "reps":        int,          # Number of completed reps so far
        "form_status": str           # A short code describing what's wrong (or "good")
    }

The form_status values you must handle:
    Squat:      "good" | "too_shallow" | "too_deep" | "adjust_hips"
    Push-Up:    "good" | "not_deep_enough" | "too_low" | "hips_sagging" | "hips_piking"
    Bicep Curl: "good" | "not_curled_enough" | "swinging"
"""


# ==============================================================================
# SECTION 1: FEEDBACK MESSAGE LOOKUP TABLE
# This dictionary maps (exercise_name, form_status) pairs to human-readable
# coaching messages. Using a lookup table is cleaner than a giant if/elif chain,
# and makes it easy to add new exercises or tweak messages later.
#
# Message format guide:
#   ✅  = good form  (shown in GREEN in the UI)
#   ⚠️  = needs fixing (shown in ORANGE in the UI)
#   ❌  = serious error (shown in RED in the UI)
# ==============================================================================

FEEDBACK_MESSAGES = {

    # --------------------------------------------------------------------------
    # SQUAT
    # Key angles to think about:
    #   - Knee angle:  ~90° at the bottom of a good squat
    #   - Hip angle:   hips should stay above knees (not collapse forward/back)
    # --------------------------------------------------------------------------
    "Squat": {
        "good":         "✅ Great squat depth — keep that form!",
        "too_shallow":  "⚠️ Go deeper — aim for 90° at the knees!",
        "too_deep":     "⚠️ Careful — you're going too deep, protect your knees!",
        "adjust_hips":  "⚠️ Keep your hips back — don't let your chest fall forward!",
    },

    # --------------------------------------------------------------------------
    # PUSH-UP
    # Key angles to think about:
    #   - Elbow angle:  ~90° at the bottom of a good push-up
    #   - Hip angle:    body should stay in a straight plank line (no sagging/piking)
    # --------------------------------------------------------------------------
    "Push-Up": {
        "good":             "✅ Perfect push-up form — keep it up!",
        "not_deep_enough":  "⚠️ Lower your chest more — aim for a 90° elbow bend!",
        "too_low":          "⚠️ You're going too low — stop when elbows hit 90°!",
        "hips_sagging":     "❌ Hips are drooping! Squeeze your core to stay flat!",
        "hips_piking":      "⚠️ Hips are too high — lower them to form a straight line!",
    },

    # --------------------------------------------------------------------------
    # BICEP CURL
    # Key angles to think about:
    #   - Elbow angle:  ~30–40° fully curled, ~160°+ fully extended
    #   - Upper arm:    should stay still against the body (no swinging)
    # --------------------------------------------------------------------------
    "Bicep Curl": {
        "good":             "✅ Nice curl — full range of motion!",
        "not_curled_enough":"⚠️ Curl higher! Bring the weight up to your shoulder!",
        "swinging":         "⚠️ Don't swing! Keep your upper arm still and control the weight!",
    },
}


# ==============================================================================
# SECTION 2: FALLBACK MESSAGE
# Used when the exercise name or form status isn't in our lookup table.
# This prevents crashes if a new exercise is added but feedback isn't yet written.
# ==============================================================================

FALLBACK_MESSAGE = "🔍 Analyzing your form..."


# ==============================================================================
# SECTION 3: THE MAIN FUNCTION
# This is the function that main.py imports and calls every frame.
# ==============================================================================

def get_feedback(analysis_result, exercise_name):
    """
    Returns a human-readable coaching message based on exercise analysis data.

    This function is called by main.py once per frame (when a pose is detected).
    It looks up the appropriate message using the exercise name and form_status
    from the analysis result.

    Args:
        analysis_result (dict): The output from an exercise analysis function.
            Expected keys:
                "angles"      (dict)  — Joint angles in degrees. Not used here
                                        directly, but available if you want to
                                        add angle-specific messages later.
                "reps"        (int)   — Number of completed reps.
                "form_status" (str)   — Short code like "good", "too_shallow", etc.

        exercise_name (str): Which exercise is selected. One of:
                             "Squat", "Push-Up", "Bicep Curl"

    Returns:
        str: A friendly coaching message string with an emoji prefix.
             Examples:
               "✅ Great squat depth — keep that form!"
               "⚠️ Go deeper — aim for 90° at the knees!"
               "❌ Hips are drooping! Squeeze your core to stay flat!"

    Example call (from main.py):
        feedback_message = get_feedback(analysis_result, "Squat")
        app_window.update_feedback(feedback_message)
    """

    # --- Step 1: Safely extract form_status from the dict ---
    # We use .get() instead of [] so we don't crash if the key is missing.
    # Default to an empty string if "form_status" isn't in the dict.
    form_status = analysis_result.get("form_status", "")

    # --- Step 2: Look up the exercise in our feedback table ---
    # FEEDBACK_MESSAGES is a dict of dicts:
    #   { "Squat": { "good": "✅ ...", "too_shallow": "⚠️ ..." }, ... }
    # .get(exercise_name, {}) returns an empty dict if exercise_name isn't found
    # — this prevents a KeyError crash for unknown exercises.
    exercise_messages = FEEDBACK_MESSAGES.get(exercise_name, {})

    # --- Step 3: Look up the specific status message ---
    # .get(form_status, FALLBACK_MESSAGE) returns the fallback string if
    # form_status isn't in the exercise's message dict.
    message = exercise_messages.get(form_status, FALLBACK_MESSAGE)

    return message


# ==============================================================================
# SECTION 4: OPTIONAL HELPER — Rep Milestone Messages
# Call this separately if you want to celebrate rep milestones in the UI.
# Not required by main.py, but a nice touch for the UX!
# ==============================================================================

def get_rep_milestone_message(reps):
    """
    Returns a motivational message when the user hits a rep milestone.
    Optional — call this from main.py or app_window.py if you want milestone popups.

    Args:
        reps (int): The current rep count.

    Returns:
        str or None: A milestone message, or None if it's not a milestone rep.

    Example:
        msg = get_rep_milestone_message(analysis_result["reps"])
        if msg:
            app_window.show_milestone(msg)  # hypothetical UI method
    """
    # Define milestones and their messages
    MILESTONES = {
        5:  "🔥 5 reps — warming up!",
        10: "💪 10 reps — halfway there!",
        15: "⚡ 15 reps — you're on fire!",
        20: "🏆 20 reps — incredible work!",
        25: "🌟 25 reps — legendary!",
    }
    # Return the milestone message if this rep count is a milestone, else None
    return MILESTONES.get(reps, None)


# ==============================================================================
# STANDALONE TEST
# Run this file directly to verify your feedback logic works:
#     python utils/feedback.py
#
# You should see a list of test results printed to the terminal.
# All tests should print the expected messages.
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  feedback.py Standalone Test")
    print("=" * 60)

    # --- Build some fake analysis_result dicts to test with ---
    # In real usage, these come from exercises/squat.py, pushup.py, etc.
    test_cases = [
        # (analysis_result dict,                       exercise_name,   expected_emoji)
        ({"angles": {}, "reps": 3, "form_status": "good"},          "Squat",      "✅"),
        ({"angles": {}, "reps": 1, "form_status": "too_shallow"},   "Squat",      "⚠️"),
        ({"angles": {}, "reps": 0, "form_status": "too_deep"},      "Squat",      "⚠️"),
        ({"angles": {}, "reps": 5, "form_status": "adjust_hips"},   "Squat",      "⚠️"),

        ({"angles": {}, "reps": 2, "form_status": "good"},          "Push-Up",    "✅"),
        ({"angles": {}, "reps": 1, "form_status": "hips_sagging"},  "Push-Up",    "❌"),
        ({"angles": {}, "reps": 1, "form_status": "hips_piking"},   "Push-Up",    "⚠️"),
        ({"angles": {}, "reps": 1, "form_status": "not_deep_enough"}, "Push-Up",  "⚠️"),
        ({"angles": {}, "reps": 1, "form_status": "too_low"},       "Push-Up",    "⚠️"),

        ({"angles": {}, "reps": 8, "form_status": "good"},          "Bicep Curl", "✅"),
        ({"angles": {}, "reps": 2, "form_status": "swinging"},      "Bicep Curl", "⚠️"),
        ({"angles": {}, "reps": 2, "form_status": "not_curled_enough"}, "Bicep Curl", "⚠️"),

        # Edge case: unknown exercise
        ({"angles": {}, "reps": 0, "form_status": "good"},          "Lunges",     "🔍"),
        # Edge case: missing form_status key
        ({"angles": {}, "reps": 0},                                 "Squat",      "🔍"),
    ]

    all_passed = True
    for i, (result, exercise, expected_start) in enumerate(test_cases):
        message = get_feedback(result, exercise)
        passed = message.startswith(expected_start)
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False
        print(f"  [{status}] {exercise:12s} | status={result.get('form_status', 'MISSING'):20s} → {message}")

    print()
    if all_passed:
        print("✅ All tests passed! feedback.py is working correctly.")
    else:
        print("❌ Some tests failed — check the output above.")

    # --- Test rep milestones ---
    print()
    print("Rep milestone test:")
    for rep_count in [1, 5, 9, 10, 15, 20, 25]:
        msg = get_rep_milestone_message(rep_count)
        print(f"  Reps={rep_count:3d} → {msg if msg else '(no milestone)'}")