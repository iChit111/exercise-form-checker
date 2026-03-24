# Exercise Form Checker

An AI-powered desktop application that analyzes workout form using real-time pose detection. This tool uses MediaPipe to detect body landmarks and provides real-time feedback on exercise form for squats, push-ups, and bicep curls.

## Features

- **Real-time Pose Detection**: Uses MediaPipe to detect 33 body landmarks in live webcam feed
- **Exercise Analysis**: Analyzes three exercises:
  - **Squats**: Checks knee angles, hip angles, and back alignment
  - **Push-ups**: Validates arm angles, body alignment, and depth
  - **Bicep Curls**: Monitors elbow and shoulder positioning
- **Live Feedback**: Displays color-coded coaching messages and form corrections
- **Rep Counter**: Automatically counts completed repetitions
- **Desktop UI**: PyQt5-based graphical interface with live video feed

## Project Structure

```
exercise-form-checker/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── pose_landmarker_full.task  # MediaPipe pose model
├── exercises/
│   ├── squat.py              # Squat form analysis logic
│   ├── pushup.py             # Push-up form analysis logic
│   └── bicep_curl.py         # Bicep curl form analysis logic
├── utils/
│   ├── pose_detector.py      # MediaPipe wrapper and pose detection
│   ├── angle_calculator.py   # Joint angle calculations
│   └── feedback.py           # Coaching messages and feedback logic
├── ui/
│   └── app_window.py         # PyQt5 desktop UI and window management
└── README.md                 # This file
```

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in `requirements.txt`:
  - mediapipe >= 0.10.30
  - opencv-python
  - numpy
  - pyqt5

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd exercise-form-checker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Select an exercise (Squat, Push-up, or Bicep Curl) from the dropdown menu

3. Click "Start" to begin form analysis

4. Position yourself in front of the webcam with good lighting

5. The application will:
   - Display your skeleton overlay
   - Show real-time feedback on your form
   - Count completed repetitions
   - Highlight form issues with coaching tips

6. Click "Stop" to end the session

## How It Works

### Pose Detection
The application uses MediaPipe's PoseLandmarker to detect 33 body keypoints in real-time. Each landmark provides:
- X/Y position (normalized 0.0-1.0)
- Z depth coordinate
- Visibility confidence score

### Exercise Analysis
Each exercise module analyzes relevant joint angles:
- **Angles are calculated** between three points (e.g., Hip → Knee → Ankle)
- **Form validation** checks if angles fall within acceptable ranges
- **Rep counting** tracks when users complete full repetitions
- **Feedback generation** provides specific coaching messages

### Real-time Feedback
The UI displays:
- Live video feed with skeleton overlay
- Color-coded feedback messages (green = good form, red = needs correction)
- Current repetition count
- Real-time form corrections

## Technical Architecture

- **MediaPipe**: AI-based pose detection and body landmark recognition
- **OpenCV**: Video capture and frame processing
- **NumPy**: Mathematical calculations for joint angles
- **PyQt5**: Desktop GUI and user interface

## Performance Considerations

- Run in good lighting conditions for best pose detection
- Ensure full body visibility in camera frame
- Recommended: webcam resolution 640x480 or higher
- Application runs pose detection on each video frame for real-time feedback

## Supported Exercises

### Squat
Analyzes knee angle (90-110° at bottom), hip angle (80-100°), and spine alignment.

### Push-up
Validates elbow angle, body alignment, and full extension/flexion range.

### Bicep Curl
Monitors elbow positioning, shoulder stability, and proper curl range of motion.

## Future Enhancements

- Additional exercises (deadlifts, lunges, shoulder press)
- Form history and progress tracking
- Video recording and playback
- Mobile app version
- Multiplayer form comparison

## Contributors

This project was developed as a collaborative exercise in AI-powered fitness technology.
