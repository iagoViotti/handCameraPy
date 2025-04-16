import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import tkinter as tk
from PIL import Image, ImageTk

# Setup snapshot directory
os.makedirs("snapshots", exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Get camera frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tkinter setup
root = tk.Tk()
root.title("Hand Rectangle Drawing")
canvas = tk.Label(root)
canvas.pack()

# Global reference for snapshot image display
snapshot_windows = []

# Mediapipe hands instance
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


def show_snapshot(image):
    top = tk.Toplevel(root)
    top.title("Snapshot")
    img = Image.fromarray(image)  # The snapshot is already in RGB format
    img_tk = ImageTk.PhotoImage(image=img)
    label = tk.Label(top, image=img_tk)
    label.image = img_tk  # Keep reference
    label.pack()
    snapshot_windows.append(top)


def update():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for both processing and display
    result = hands.process(rgb)

    points = []
    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        left_hand = result.multi_hand_landmarks[0]
        right_hand = result.multi_hand_landmarks[1]

        left_thumb = left_hand.landmark[4]
        left_index = left_hand.landmark[8]
        right_thumb = right_hand.landmark[4]
        right_index = right_hand.landmark[8]

        for hand_landmarks in result.multi_hand_landmarks:
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            h, w, _ = frame.shape
            thumb_coords = (int(thumb.x * w), int(thumb.y * h))
            index_coords = (int(index.x * w), int(index.y * h))

            mid_x = (thumb_coords[0] + index_coords[0]) // 2
            mid_y = (thumb_coords[1] + index_coords[1]) // 2

            cv2.circle(rgb, thumb_coords, 4, (255, 255, 255), 1, 1)
            cv2.circle(rgb, index_coords, 4, (255, 255, 255), 1, 1)

            points.append((mid_x, mid_y))

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # Dark overlay logic
            overlay = rgb.copy()
            overlay[:] = (0, 0, 0)
            mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            mask_3ch = cv2.merge([mask, mask, mask])
            dark_frame = cv2.addWeighted(rgb, 0.4, overlay, 0.6, 0)
            rgb = np.where(mask_3ch == 255, rgb, dark_frame)

            # Snapshot trigger logic
        if (abs(left_thumb.x - left_index.x) < 0.04 and abs(left_thumb.y - left_index.y) < 0.04 and
            abs(right_thumb.x - right_index.x) < 0.04 and abs(right_thumb.y - right_index.y) < 0.04):
            snapshot = rgb[y1:y2, x1:x2]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshots/snapshot_{timestamp}.png"
            cv2.imwrite(path, cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            print(f"Snapshot saved: {path}")
            show_snapshot(snapshot)

    # Convert to ImageTk and display
    img = Image.fromarray(rgb)  # Use RGB frame for display
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)

    root.after(10, update)


# Start update loop
update()
root.mainloop()

# Cleanup
cap.release()
