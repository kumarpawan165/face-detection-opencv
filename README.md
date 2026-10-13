# Real-Time Face Detection using OpenCV

Detects faces in real time from a webcam using OpenCV and a Haar Cascade classifier.

**Live demo:** https://kumarpawan165.github.io/face-detection-opencv/

Two versions are included:
- **Web version** (`index.html`): runs in the browser with OpenCV.js, with sliders to tune the detector
- **Python version** (`face_detection.py`): desktop webcam version using `opencv-python`

## How it works
1. Captures frames from the webcam
2. Converts each frame to grayscale (preprocessing)
3. Detects faces with a pre-trained Haar Cascade classifier (`detectMultiScale`)
4. Draws a bounding box around every detected face

## Tech
Python, OpenCV, OpenCV.js, JavaScript, HTML/CSS

## Run the Python version
```bash
pip install opencv-python
python face_detection.py
```
Press `q` to quit.

## Parameters to experiment with
- `scaleFactor` (1.1 vs 1.3): lower is more thorough but slower
- `minNeighbors` (2 vs 5): lower gives more false detections, higher can miss real faces

## Limitations
Haar cascades can miss tilted or side-facing faces and struggle in low light. Deep learning models like YOLO or MTCNN would be more accurate.

## Possible next steps
- Face recognition (who, not just where)
- Eye and smile detection
- A DNN-based detector
