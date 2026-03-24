Exercise Form Checker

A real-time computer vision application that helps users maintain correct form during various exercises. Using MediaPipe Pose Landmarker, the app analyzes body movements through a webcam and provides instant feedback to prevent injury and maximize workout efficiency.
🚀 Features

    Real-time Pose Tracking: Utilizes the Google MediaPipe pose_landmarker_full.task for high-accuracy body tracking.

    Multiple Exercises: Built-in logic for checking form on various exercises (e.g., Squats, Pushups, Bicep Curls).

    Visual Feedback: Interactive UI that highlights joint angles and provides corrective cues.

    Rep Counter: Automatically tracks and counts completed repetitions.

🛠️ Tech Stack

    Python 3.x

    MediaPipe: For human pose estimation.

    OpenCV: For video processing and image manipulation.

    NumPy: For calculating joint angles and geometric logic.

📦 Installation

    Clone the repository:
    Bash

    git clone https://github.com/iChit111/exercise-form-checker.git
    cd exercise-form-checker

    Create a virtual environment (optional but recommended):
    Bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies:
    Bash

    pip install -r requirements.txt

🖥️ Usage

Run the main application:
Bash

python main.py

    Ensure your webcam is connected and you are in a well-lit area.

    Step back so your entire body (or the relevant limbs) is visible in the frame.

    Select an exercise from the menu and start your set!

📂 Project Structure

    main.py: The entry point of the application.

    exercises/: Contains logic for specific exercise form validation (angles, thresholds).

    ui/: Handles the graphical overlays and user interface components.

    utils/: Helper functions for mathematical calculations (e.g., calculating angles between landmarks).

    pose_landmarker_full.task: The pre-trained MediaPipe model bundle.

🤝 Contributing

Contributions are welcome! If you'd like to add a new exercise or improve the detection logic:

    Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

    Open a Pull Request
