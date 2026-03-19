"""
ui/app_window.py — Main Desktop Window (PyQt5)
===============================================
YOUR ROLE: GUI / App Window
This file defines the main window the user sees when they launch the app.
It handles:
    - Displaying the live webcam feed
    - A dropdown to pick Squat / Push-Up / Bicep Curl
    - A feedback panel that shows coaching messages (color-coded)
    - A rep counter display
    - A Start/Stop button
    - A QTimer loop that drives the per-frame processing

HOW PyQt5 WORKS (quick primer for beginners):
----------------------------------------------
PyQt5 is a GUI toolkit. Everything in the window is a "widget" — a button,
a label, a video display, etc. Widgets are arranged using "layouts" (like
rows and columns). Here's the mental model:

    QMainWindow        ← the outer shell / window frame
      └── central_widget  ← an invisible container inside the window
            └── main_layout (QVBoxLayout = vertical stack)
                  ├── top_bar (exercise picker + buttons)   ← row 1
                  ├── video_label  (webcam feed image)      ← row 2 (biggest)
                  ├── feedback_label (coaching text)        ← row 3
                  └── rep_label (rep counter)               ← row 4

PyQt5 is EVENT-DRIVEN: instead of a while loop, you respond to "signals"
(user clicks a button → a signal fires → your function runs).
The QTimer fires a signal every N milliseconds → we use that for the frame loop.
"""

import sys                          # Needed for QApplication and sys.exit()
import numpy as np                  # For type hinting (OpenCV frames are numpy arrays)
import cv2                          # For color conversion (BGR → RGB) of webcam frames

# --- PyQt5 Imports ---
# PyQt5 is organised into modules. Here's what each one provides:
from PyQt5.QtWidgets import (
    QApplication,    # The application object — every PyQt5 app needs exactly one
    QMainWindow,     # The main window shell (has title bar, menu bar, etc.)
    QWidget,         # A plain container widget (used as the "central widget")
    QLabel,          # Displays text OR images — we use it for both!
    QComboBox,       # A dropdown selector widget
    QPushButton,     # A clickable button
    QVBoxLayout,     # Vertical layout — stacks widgets top-to-bottom
    QHBoxLayout,     # Horizontal layout — arranges widgets left-to-right
    QFrame,          # A styled container (used for the feedback panel border)
    QSizePolicy,     # Controls how widgets grow/shrink when the window resizes
)
from PyQt5.QtCore import (
    Qt,              # Constants like Qt.AlignCenter, Qt.KeepAspectRatio
    QTimer,          # Fires a signal at regular intervals (our frame loop engine)
)
from PyQt5.QtGui import (
    QImage,          # Wraps raw pixel data so PyQt5 can understand it
    QPixmap,         # A displayable image object — used by QLabel to show frames
    QFont,           # For customizing text fonts
    QColor,          # For programmatic color handling
    QPalette,        # For setting widget background/foreground colors
)


# ==============================================================================
# SECTION 1: STYLE CONSTANTS
# All colors and fonts defined in one place. Easy to update the whole look
# by changing values here rather than hunting through the code.
# ==============================================================================

# Color palette — dark gym-app aesthetic
COLOR_BG_DARK       = "#0D0F14"   # Deep dark background (almost black)
COLOR_BG_PANEL      = "#161A23"   # Slightly lighter panel background
COLOR_BG_CARD       = "#1E2330"   # Card / section background
COLOR_ACCENT        = "#00E5A0"   # Vivid mint-green accent (primary brand color)
COLOR_ACCENT_DIM    = "#00A870"   # Dimmed accent for hover states
COLOR_TEXT_PRIMARY  = "#EAEDF5"   # Main white text
COLOR_TEXT_MUTED    = "#6B748A"   # Dimmed secondary text
COLOR_BORDER        = "#2A3045"   # Subtle border color
COLOR_GOOD          = "#22C55E"   # Green — good form
COLOR_WARN          = "#F59E0B"   # Amber — needs adjustment
COLOR_ERROR         = "#EF4444"   # Red — serious form issue

# Font sizes
FONT_TITLE          = 20
FONT_EXERCISE_LABEL = 12
FONT_FEEDBACK       = 16
FONT_REPS           = 48
FONT_REPS_LABEL     = 11

# Timer interval: 33ms ≈ 30 frames per second
FRAME_INTERVAL_MS   = 33


# ==============================================================================
# SECTION 2: THE APPWINDOW CLASS
# ==============================================================================

class AppWindow(QMainWindow):
    """
    The main application window for Exercise Form Checker.

    This class is responsible for ALL visual output:
        - Showing the webcam feed
        - Showing coaching feedback messages (color-coded)
        - Showing the rep counter
        - Letting the user choose an exercise from a dropdown
        - Running the per-frame callback via a QTimer

    How main.py uses this class:
        app_window = AppWindow(title="Exercise Form Checker", exercises=["Squat", ...])
        app_window.show()
        app_window.set_frame_callback(lambda: run_app(app_window, pose_detector, webcam))
    """

    def __init__(self, title="Exercise Form Checker", exercises=None):
        """
        Builds the window and all its widgets.

        Args:
            title (str): The text shown in the window's title bar.
            exercises (list): List of exercise name strings for the dropdown.
                              e.g. ["Squat", "Push-Up", "Bicep Curl"]
                              If None, uses a default list.
        """
        # --- Always call the parent class __init__ first ---
        # QMainWindow needs to initialize itself before we can use it.
        super().__init__()

        # --- Store the exercise list (use a default if none provided) ---
        if exercises is None:
            exercises = ["Squat", "Push-Up", "Bicep Curl"]
        self.exercises = exercises

        # --- Internal state ---
        self._is_running = True    # Tracks whether the timer is running (Start/Stop)
        self._timer = None         # Will hold our QTimer instance

        # --- Build the window ---
        self._setup_window(title)
        self._build_ui()
        self._apply_styles()

    # --------------------------------------------------------------------------
    # WINDOW SETUP
    # --------------------------------------------------------------------------

    def _setup_window(self, title):
        """
        Configures the basic window properties (title, size, background color).
        """
        self.setWindowTitle(title)

        # Set the minimum size so the UI doesn't get squished below a readable size
        self.setMinimumSize(900, 680)

        # Resize to a comfortable starting size
        self.resize(1100, 760)

        # Set the overall background color of the window using a stylesheet.
        # Qt stylesheets work like CSS — same property:value syntax.
        self.setStyleSheet(f"background-color: {COLOR_BG_DARK};")

    # --------------------------------------------------------------------------
    # UI CONSTRUCTION
    # Build all widgets and lay them out in the window.
    # --------------------------------------------------------------------------

    def _build_ui(self):
        """
        Creates all the widgets and arranges them in the window layout.

        Layout hierarchy:
            central_widget (QWidget)
              └── main_layout (QVBoxLayout — vertical stack)
                    ├── _build_top_bar()      ← Exercise picker + Start/Stop
                    ├── video_label           ← Live webcam feed
                    └── _build_bottom_panel() ← Feedback + rep counter
        """

        # --- Create the central widget ---
        # QMainWindow requires a "central widget" — it's the main content area.
        # We use a plain QWidget and attach our layout to it.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # --- Main vertical layout ---
        # QVBoxLayout stacks child widgets vertically, top to bottom.
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)   # Padding around the edges
        main_layout.setSpacing(12)                         # Space between rows

        # --- Row 1: Top bar (exercise selector + buttons) ---
        top_bar = self._build_top_bar()
        main_layout.addWidget(top_bar)

        # --- Row 2: Webcam feed (largest element, takes all remaining space) ---
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)     # Center the image in the label
        self.video_label.setMinimumHeight(400)

        # setSizePolicy controls how the widget expands when the window resizes.
        # Expanding in both directions means it'll take as much space as possible.
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Placeholder text shown before the webcam starts
        self.video_label.setText("📷  Webcam feed will appear here")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_TEXT_MUTED};
                border: 2px dashed {COLOR_BORDER};
                border-radius: 12px;
                font-size: 16px;
            }}
        """)

        # addWidget(widget, stretch) — stretch=1 means "give me extra space"
        main_layout.addWidget(self.video_label, stretch=1)

        # --- Row 3: Bottom panel (feedback + rep counter) ---
        bottom_panel = self._build_bottom_panel()
        main_layout.addWidget(bottom_panel)

    def _build_top_bar(self):
        """
        Builds the top control bar containing:
            - App title / logo text (left)
            - Exercise dropdown selector (center)
            - Start/Stop button (right)

        Returns:
            QWidget: The fully constructed top bar widget.
        """
        # Container widget with a horizontal layout
        bar = QWidget()
        bar.setFixedHeight(64)
        bar.setStyleSheet(f"""
            QWidget {{
                background-color: {COLOR_BG_PANEL};
                border-radius: 10px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        # --- App title (left side) ---
        title_label = QLabel("⚡ FormCheck")
        title_label.setStyleSheet(f"""
            font-size: {FONT_TITLE}px;
            font-weight: 800;
            color: {COLOR_ACCENT};
            border: none;
            background: transparent;
        """)
        layout.addWidget(title_label)

        # addStretch() pushes everything after it to the right.
        # Think of it like a rubber spacer between widgets.
        layout.addStretch()

        # --- Exercise label ---
        ex_label = QLabel("Exercise:")
        ex_label.setStyleSheet(f"""
            font-size: {FONT_EXERCISE_LABEL}px;
            color: {COLOR_TEXT_MUTED};
            border: none;
            background: transparent;
        """)
        layout.addWidget(ex_label)

        # --- Exercise dropdown (QComboBox) ---
        # QComboBox is a dropdown selector. addItems() populates it with choices.
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(self.exercises)
        self.exercise_combo.setFixedWidth(160)
        self.exercise_combo.setFixedHeight(36)
        self.exercise_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_TEXT_PRIMARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 4px 12px;
                font-size: 13px;
                font-weight: 600;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_TEXT_PRIMARY};
                selection-background-color: {COLOR_ACCENT};
                selection-color: #000000;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        layout.addWidget(self.exercise_combo)

        # --- Start/Stop button ---
        self.start_stop_btn = QPushButton("⏸  Pause")
        self.start_stop_btn.setFixedSize(120, 36)
        self.start_stop_btn.setCursor(Qt.PointingHandCursor)   # Show hand cursor on hover
        self.start_stop_btn.setStyleSheet(self._btn_style_active())

        # Connect the button's "clicked" signal to our toggle method.
        # Signals & slots: when the button is clicked, PyQt5 emits the "clicked"
        # signal. .connect() says "when that happens, call this function".
        self.start_stop_btn.clicked.connect(self._toggle_running)
        layout.addWidget(self.start_stop_btn)

        return bar

    def _build_bottom_panel(self):
        """
        Builds the bottom info panel containing:
            - Feedback message label (left side, takes most space)
            - Rep counter display (right side)

        Returns:
            QWidget: The fully constructed bottom panel widget.
        """
        panel = QWidget()
        panel.setFixedHeight(110)
        panel.setStyleSheet(f"""
            QWidget {{
                background-color: {COLOR_BG_PANEL};
                border-radius: 10px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(20)

        # --- Left side: Feedback section ---
        feedback_section = QWidget()
        feedback_section.setStyleSheet("background: transparent; border: none;")
        feedback_v = QVBoxLayout(feedback_section)
        feedback_v.setContentsMargins(0, 0, 0, 0)
        feedback_v.setSpacing(4)

        feedback_title = QLabel("COACHING FEEDBACK")
        feedback_title.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            color: {COLOR_TEXT_MUTED};
            background: transparent;
            border: none;
        """)
        feedback_v.addWidget(feedback_title)

        # Main feedback label — this is what gets updated every frame
        self.feedback_label = QLabel("🔍 Waiting for pose detection...")
        self.feedback_label.setWordWrap(True)      # Wrap text if it's too long
        self.feedback_label.setStyleSheet(f"""
            font-size: {FONT_FEEDBACK}px;
            font-weight: 600;
            color: {COLOR_TEXT_PRIMARY};
            background: transparent;
            border: none;
        """)
        feedback_v.addWidget(self.feedback_label)
        feedback_v.addStretch()

        layout.addWidget(feedback_section, stretch=1)

        # --- Divider line between feedback and rep counter ---
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)   # Vertical line
        divider.setStyleSheet(f"color: {COLOR_BORDER}; background: {COLOR_BORDER};")
        divider.setFixedWidth(1)
        layout.addWidget(divider)

        # --- Right side: Rep counter ---
        rep_section = QWidget()
        rep_section.setFixedWidth(150)
        rep_section.setStyleSheet("background: transparent; border: none;")
        rep_v = QVBoxLayout(rep_section)
        rep_v.setContentsMargins(16, 0, 0, 0)
        rep_v.setSpacing(0)
        rep_v.setAlignment(Qt.AlignCenter)

        rep_title = QLabel("REPS")
        rep_title.setAlignment(Qt.AlignCenter)
        rep_title.setStyleSheet(f"""
            font-size: {FONT_REPS_LABEL}px;
            font-weight: 700;
            letter-spacing: 3px;
            color: {COLOR_TEXT_MUTED};
            background: transparent;
            border: none;
        """)
        rep_v.addWidget(rep_title)

        # The big rep number — updated by update_rep_count()
        self.rep_label = QLabel("0")
        self.rep_label.setAlignment(Qt.AlignCenter)
        self.rep_label.setStyleSheet(f"""
            font-size: {FONT_REPS}px;
            font-weight: 900;
            color: {COLOR_ACCENT};
            background: transparent;
            border: none;
        """)
        rep_v.addWidget(self.rep_label)

        layout.addWidget(rep_section)

        return panel

    # --------------------------------------------------------------------------
    # STYLING HELPERS
    # --------------------------------------------------------------------------

    def _apply_styles(self):
        """
        Applies any global styles that need to be set after widgets are built.
        (Most styles are set inline above, but this is a good place for anything
        that needs to reference the built widgets.)
        """
        pass  # Currently all styles are handled inline in the build methods

    def _btn_style_active(self):
        """Returns the stylesheet for the button in its ACTIVE (running) state."""
        return f"""
            QPushButton {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_ACCENT};
                border: 1.5px solid {COLOR_ACCENT};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT};
                color: #000000;
            }}
        """

    def _btn_style_paused(self):
        """Returns the stylesheet for the button in its PAUSED state."""
        return f"""
            QPushButton {{
                background-color: {COLOR_ACCENT};
                color: #000000;
                border: 1.5px solid {COLOR_ACCENT};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT_DIM};
                color: #000000;
            }}
        """

    # --------------------------------------------------------------------------
    # PUBLIC API METHODS
    # These are the methods that main.py calls every frame.
    # --------------------------------------------------------------------------

    def show(self):
        """
        Displays the window on screen.
        Overrides QMainWindow.show() — just calls the parent's version.
        This is here so it's clearly documented in the interface.
        """
        super().show()

    def set_frame_callback(self, fn):
        """
        Sets up a QTimer that calls fn() approximately 30 times per second.

        Why QTimer instead of a while loop?
        A while loop would BLOCK the event loop, freezing the window — it couldn't
        respond to clicks, repaints, or anything else. QTimer fires a "tick" signal
        at regular intervals WITHOUT blocking, so the UI stays responsive.

        Args:
            fn (callable): The function to call each frame.
                           In main.py this is: lambda: run_app(app_window, pose_detector, webcam)

        How this works:
            1. Create a QTimer object
            2. Connect its "timeout" signal to our fn
            3. Start the timer with an interval of FRAME_INTERVAL_MS milliseconds
            4. Every FRAME_INTERVAL_MS ms, the timer fires, which calls fn()
        """
        self._timer = QTimer()

        # Connect the timer's timeout signal to the callback function.
        # Each time the timer "ticks", it emits the timeout signal,
        # which calls fn() automatically.
        self._timer.timeout.connect(fn)

        # Start the timer. It will now fire every FRAME_INTERVAL_MS milliseconds.
        # 33ms ≈ 30 fps
        self._timer.start(FRAME_INTERVAL_MS)

        print(f"[AppWindow] Frame timer started — {1000 // FRAME_INTERVAL_MS} fps target.")

    def get_selected_exercise(self):
        """
        Returns the name of the currently selected exercise from the dropdown.

        Returns:
            str: One of the exercise names, e.g. "Squat", "Push-Up", "Bicep Curl"

        Example:
            exercise = app_window.get_selected_exercise()
            # → "Squat"
        """
        # QComboBox.currentText() returns the text of the currently selected item
        return self.exercise_combo.currentText()

    def update_video_feed(self, frame):
        """
        Displays a new webcam frame in the video area.

        IMPORTANT: OpenCV frames are in BGR color format.
        PyQt5 expects RGB. We MUST convert, otherwise colors are wrong
        (reds and blues will be swapped).

        Args:
            frame (numpy.ndarray): A BGR image from OpenCV.
                                   Shape: (height, width, 3)

        How the conversion works:
            1. cv2.cvtColor converts BGR → RGB
            2. QImage wraps the raw pixel data in a Qt-understandable object
            3. QPixmap converts QImage to a displayable pixmap
            4. .scaled() resizes the pixmap to fit the label without stretching
            5. QLabel.setPixmap() displays it
        """
        if frame is None:
            return

        # Step 1: Convert BGR (OpenCV default) → RGB (Qt default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 2: Get frame dimensions
        h, w, ch = rgb_frame.shape

        # Step 3: Calculate bytes per line (needed by QImage)
        # Each pixel has 3 channels (R, G, B), so bytes_per_line = width * 3
        bytes_per_line = ch * w

        # Step 4: Wrap the numpy array in a QImage
        # QImage.Format_RGB888 means 8 bits per channel, RGB order
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Step 5: Convert QImage → QPixmap (what QLabel can actually display)
        pixmap = QPixmap.fromImage(qt_image)

        # Step 6: Scale the pixmap to fit the label, keeping the aspect ratio.
        # Qt.KeepAspectRatio means the image won't stretch or squish.
        # Qt.SmoothTransformation gives us better quality when scaling down.
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Step 7: Set the pixmap on the label to display it
        self.video_label.setPixmap(scaled_pixmap)

        # Clear the placeholder text style now that we have real video
        if self.video_label.styleSheet():
            self.video_label.setStyleSheet(
                f"background-color: {COLOR_BG_CARD}; border-radius: 12px;"
            )

    def update_feedback(self, message):
        """
        Updates the coaching feedback text shown in the bottom panel.

        Also changes the text color based on the emoji prefix:
            ✅ → green   (good form)
            ⚠️ → amber   (needs adjustment)
            ❌ → red     (serious issue)
            🔍 → muted   (waiting/neutral)

        Args:
            message (str): A human-readable coaching message.
                           e.g. "✅ Great squat depth — keep that form!"
        """
        if message is None:
            return

        # Determine color based on the first character(s) of the message
        if message.startswith("✅"):
            color = COLOR_GOOD
        elif message.startswith("⚠️"):
            color = COLOR_WARN
        elif message.startswith("❌"):
            color = COLOR_ERROR
        else:
            color = COLOR_TEXT_MUTED   # Neutral for "waiting" messages

        self.feedback_label.setText(message)
        self.feedback_label.setStyleSheet(f"""
            font-size: {FONT_FEEDBACK}px;
            font-weight: 600;
            color: {color};
            background: transparent;
            border: none;
        """)

    def update_rep_count(self, count):
        """
        Updates the large rep counter number displayed on the right side.

        Args:
            count (int): The current rep count to display.

        Example:
            app_window.update_rep_count(7)  # Shows "7" in the rep counter
        """
        self.rep_label.setText(str(count))

    # --------------------------------------------------------------------------
    # INTERNAL METHODS (private — not called by main.py)
    # --------------------------------------------------------------------------

    def _toggle_running(self):
        """
        Called when the user clicks the Start/Stop button.
        Pauses or resumes the frame timer.
        """
        if self._is_running:
            # Currently running → pause
            if self._timer:
                self._timer.stop()
            self._is_running = False
            self.start_stop_btn.setText("▶  Resume")
            self.start_stop_btn.setStyleSheet(self._btn_style_paused())
            self.update_feedback("⏸  Detection paused — click Resume to continue.")
        else:
            # Currently paused → resume
            if self._timer:
                self._timer.start(FRAME_INTERVAL_MS)
            self._is_running = True
            self.start_stop_btn.setText("⏸  Pause")
            self.start_stop_btn.setStyleSheet(self._btn_style_active())
            self.update_feedback("🔍 Waiting for pose detection...")

    def closeEvent(self, event):
        """
        Called automatically by PyQt5 when the user closes the window (X button).
        We stop the timer here to avoid errors from the timer firing after the
        window has already been destroyed.

        Args:
            event: The close event (passed automatically by Qt — we just accept it).
        """
        print("[AppWindow] Window closing — stopping frame timer.")
        if self._timer:
            self._timer.stop()
        event.accept()   # Tell Qt it's okay to close the window


# ==============================================================================
# STANDALONE TEST
# Run this file directly to check that the window opens correctly:
#     python ui/app_window.py
#
# A window should appear with:
#   - A dark top bar with "⚡ FormCheck", a dropdown, and a Pause button
#   - A placeholder video area (no real webcam needed)
#   - A feedback panel at the bottom
#   - A rep counter showing "0"
#
# The feedback text will cycle through demo messages automatically so you can
# see the color-coding working without needing any real exercise data.
# Press the Pause button to stop, Resume to start again. Close the window to exit.
# ==============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  AppWindow Standalone Test")
    print("  No webcam needed — uses a dummy colored frame.")
    print("  Close the window to exit.")
    print("=" * 60)

    # Every PyQt5 app needs exactly one QApplication instance
    app = QApplication(sys.argv)

    # Create our window with some test exercises
    window = AppWindow(
        title="Exercise Form Checker — UI Test",
        exercises=["Squat", "Push-Up", "Bicep Curl"]
    )
    window.show()

    # --- Demo data to cycle through ---
    DEMO_MESSAGES = [
        "✅ Great squat depth — keep that form!",
        "⚠️ Go deeper — aim for 90° at the knees!",
        "⚠️ Keep your hips back — don't let your chest fall forward!",
        "❌ Hips are drooping! Squeeze your core to stay flat!",
        "🔍 Waiting for pose detection...",
        "✅ Perfect push-up form — keep it up!",
        "⚠️ Don't swing! Keep your upper arm still!",
    ]
    demo_index = [0]      # Use a list so the inner function can modify it
    demo_reps   = [0]

    def demo_frame_callback():
        """
        Simulates what run_app() does each frame, but without a real webcam.
        Creates a colored rectangle as a fake "video frame".
        """
        # Create a fake BGR frame (a solid colored rectangle)
        # numpy.zeros creates a black image; we fill it with a color
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame[:, :] = (30, 40, 55)   # BGR dark blue-gray

        # Draw some fake "skeleton" lines so it looks like something is happening
        center_x, center_y = 320, 240
        cv2.circle(fake_frame, (center_x, center_y - 80), 30, (0, 229, 160), 2)  # head
        cv2.line(fake_frame, (center_x, center_y - 50), (center_x, center_y + 60), (0, 229, 160), 2)  # body
        cv2.line(fake_frame, (center_x, center_y), (center_x - 60, center_y + 30), (0, 229, 160), 2)  # left arm
        cv2.line(fake_frame, (center_x, center_y), (center_x + 60, center_y + 30), (0, 229, 160), 2)  # right arm
        cv2.line(fake_frame, (center_x, center_y + 60), (center_x - 40, center_y + 130), (0, 229, 160), 2)  # left leg
        cv2.line(fake_frame, (center_x, center_y + 60), (center_x + 40, center_y + 130), (0, 229, 160), 2)  # right leg

        # Add a label
        cv2.putText(fake_frame, "DEMO MODE — No Webcam", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 120, 140), 2)

        # Update the video display with our fake frame
        window.update_video_feed(fake_frame)

        # Cycle through demo feedback messages (change every ~90 frames ≈ 3 seconds)
        frame_modulo = 90
        current_frame = getattr(demo_frame_callback, "_frame_count", 0)
        demo_frame_callback._frame_count = current_frame + 1

        if current_frame % frame_modulo == 0:
            msg = DEMO_MESSAGES[demo_index[0] % len(DEMO_MESSAGES)]
            window.update_feedback(msg)
            demo_index[0] += 1

        # Increment rep count every 150 frames (~5 seconds)
        if current_frame % 150 == 0 and current_frame > 0:
            demo_reps[0] += 1
            window.update_rep_count(demo_reps[0])

    # Set the demo function as our frame callback
    window.set_frame_callback(demo_frame_callback)

    print("[TEST] Window is open. Watch for cycling feedback messages and rep counter!")
    print("[TEST] Try the exercise dropdown and Pause/Resume button.")

    # qt_app.exec_() starts the event loop — blocks here until the window closes
    exit_code = app.exec_()
    print("[TEST] Window closed. Test complete!")
    sys.exit(exit_code)