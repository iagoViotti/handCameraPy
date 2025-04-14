import cv2
import mediapipe as mp
import numpy as np
import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        points = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                h, w, _ = frame.shape
                thumb_coords = (int(thumb.x * w), int(thumb.y * h))
                index_coords = (int(index.x * w), int(index.y * h))

                mid_x = (thumb_coords[0] + index_coords[0]) // 2
                mid_y = (thumb_coords[1] + index_coords[1]) // 2

                points.append((mid_x, mid_y))

                # Draw the landmarks and lines
                cv2.circle(frame, thumb_coords, 4, (255, 255, 255), 1, 1)
                cv2.circle(frame, index_coords, 4, (255, 255, 255), 1, 1)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            h, w, _ = frame.shape

            # Create a dark overlay
            overlay = frame.copy()
            overlay[:] = (0, 0, 0)  # fully black

            # Create a binary mask with a white rectangle (area to keep clear)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # Convert to 3 channel mask for colored images
            mask_3ch = cv2.merge([mask, mask, mask])

            # Darken the frame using addWeighted
            dark_frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

            # Combine: keep original where mask is white, darkened where black
            frame = np.where(mask_3ch == 255, frame, dark_frame)

        cv2.imshow('Hand Rectangle Drawing', frame)

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            left_hand = result.multi_hand_landmarks[0]
            right_hand = result.multi_hand_landmarks[1]

            left_thumb = left_hand.landmark[4]
            left_index = left_hand.landmark[8]
            right_thumb = right_hand.landmark[4]
            right_index = right_hand.landmark[8]

            if (abs(left_thumb.x - left_index.x) < 0.04 and abs(left_thumb.y - left_index.y) < 0.04 and
                    abs(right_thumb.x - right_index.x) < 0.04 and abs(right_thumb.y - right_index.y) < 0.04):
                cropped_image = frame[y1:y2, x1:x2]
                timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
                filename = f"snapshots/{timestamp}.png"
                cv2.imwrite(filename, cropped_image)
                print(f"📸 Snapshot taken: {filename}")
                cv2.imshow(filename, cropped_image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
