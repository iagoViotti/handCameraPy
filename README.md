# TouchDesigner Bootlag

This repository is a **study project** focused on real-time hand tracking and interaction using Python, OpenCV, and Mediapipe. The goal is to explore computer vision techniques and experiment with hand gesture recognition and dynamic visual effects.

## Features

- **Real-Time Hand Tracking**: Uses Mediapipe's hand landmarks to detect and track hands in a webcam feed.
- **Dynamic Visual Effects**: Draws rectangles and highlights areas of interest based on hand gestures.
- **Snapshot Functionality**: Captures and saves cropped images when specific hand gestures are detected.
- **Interactive Visualization**: Displays the processed video feed with overlays and effects.

## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- OpenCV
- Mediapipe
- NumPy

You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```

## How It Works

1. The program captures video from the webcam using OpenCV.
2. Mediapipe detects hand landmarks and provides normalized coordinates for key points (e.g., thumb and index finger).
3. The program calculates the midpoint between the thumb and index finger for each hand.
4. If two hands are detected, a rectangle is drawn between the midpoints of the two hands.
5. When both hands' thumb and index fingers are close enough, a snapshot of the rectangle area is saved in the snapshots directory.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/touchDesignerBootlag.git
   cd touchDesignerBootlag
   ```

2. Run the program:
   ```bash
   python main.py
   ```

3. Interact with the webcam feed:
   - Move your hands to see the dynamic rectangle effect.
   - Bring the thumb and index fingers of both hands close together to capture a snapshot.

4. Press `ESC` to exit the program.

## Directory Structure

```
touchDesignerBootlag/
├── main.py          # Main script for hand tracking and interaction
├── snapshots/       # Directory where snapshots are saved
├── .gitignore       # Git ignore file for excluding unnecessary files
```

## Notes

- This repository is for **educational purposes** and is not intended for production use.
- The snapshots directory is ignored by Git to prevent cluttering the repository with generated images.

## Future Improvements

- Add gesture-based controls for more interactive features.
- Optimize performance for smoother real-time processing.
- Explore additional Mediapipe functionalities, such as pose or face detection.

## License

This project is open-source and available for study purposes. Feel free to modify and experiment with the code.

---

Happy coding! 🎉