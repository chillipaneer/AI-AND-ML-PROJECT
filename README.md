# Computer Vision AIML Project

This project demonstrates various computer vision applications using OpenCV and MediaPipe libraries. It includes multiple scripts for different computer vision tasks.

## Project Components

- **air.py**: Basic webcam application that captures and displays video feed
- **improvair.py**: Enhanced version with additional computer vision features
- **alt.py**: Alternative implementation with different computer vision techniques

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone or download this repository
2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.\.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install required packages:
   ```
   pip install opencv-python mediapipe numpy
   ```

## Running the Applications

For convenience, batch files are provided to run each application with the virtual environment activated:

- **run_air.bat**: Runs the basic webcam application
- **run_improvair.bat**: Runs the enhanced version
- **run_alt.bat**: Runs the alternative implementation
- **run_newnew.bat**: Runs the advanced air drawing system (newnew.py)

Simply double-click on the desired batch file or run it from the command line.

## Usage

- Press 'ESC' or 'q' to exit any of the applications
- You can also close the application window using the 'X' button

## Project Structure

```
air/
├── .venv/                  # Virtual environment
├── .vscode/                # VSCode settings
├── air.py                  # Basic webcam application
├── improvair.py            # Enhanced version
├── alt.py                  # Alternative implementation
├── run_air.bat             # Batch file to run air.py
├── run_improvair.bat       # Batch file to run improvair.py
├── run_alt.bat             # Batch file to run alt.py
└── README.md               # This file
```

## Dependencies

- OpenCV: Computer vision library for image and video processing
- MediaPipe: Google's framework for building multimodal applied ML pipelines
- NumPy: Library for numerical computations in Python