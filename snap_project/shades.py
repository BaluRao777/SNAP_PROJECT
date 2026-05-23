import os
import platform
import sys
import time

import cv2
import numpy as np
from my_CNN_model import load_my_CNN_model

from paths import FACE_CASCADE, FILTER_IMAGES, IMAGES_DIR, MODEL_WEIGHTS


def _camera_help_message():
    if platform.system() != 'Darwin':
        return (
            'Could not open the webcam. Close other apps using the camera '
            'or try another index: python shades.py 1'
        )
    return (
        'Could not open the webcam on macOS.\n\n'
        '1. When prompted, click Allow for camera access.\n'
        '2. Open System Settings → Privacy & Security → Camera.\n'
        '3. Enable the app you use to run Python (e.g. Terminal, iTerm, or Cursor).\n'
        '4. If Python is listed, enable it too.\n'
        '5. Quit and reopen that app, then run: python shades.py\n\n'
        'Tip: Running from the macOS Terminal app often fixes permission issues.\n'
        'External camera: python shades.py 1'
    )


def open_webcam(device_index=0, retries=5, retry_delay=2.0):
    """Open webcam; on macOS prefer AVFoundation and allow time for permission dialog."""
    backends = [cv2.CAP_ANY]
    if platform.system() == 'Darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    for attempt in range(retries):
        for backend in backends:
            cap = (
                cv2.VideoCapture(device_index, backend)
                if backend != cv2.CAP_ANY
                else cv2.VideoCapture(device_index)
            )
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.1)
            cap.release()

        if attempt == 0:
            print('Waiting for camera access (check for a permission popup)...')
        elif attempt < retries - 1:
            print(f'Retrying camera ({attempt + 1}/{retries})...')
        time.sleep(retry_delay)

    return None

# Load the model built in the previous step
my_model = load_my_CNN_model(MODEL_WEIGHTS)

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
kernel = np.ones((5, 5), np.uint8)

# Define filters (PNG/JPG assets in images/)
filters = [os.path.join(IMAGES_DIR, name) for name in FILTER_IMAGES]
filterIndex = 0

# Load the video (optional CLI arg: camera index, default 0)
device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
camera = open_webcam(device_index)
if camera is None:
    print(_camera_help_message())
    sys.exit(1)

# Keep looping
while True:
    (grabbed, frame) = camera.read()
    if not grabbed or frame is None:
        print('Lost camera frame. Check cable/permissions and try again.')
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#colur to gray

    # showing text
    cv2.putText(frame, "Press 'n' for NEXT FILTER, 'p' for PREVIOUS FILTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText( frame, "Note : show Gesture like Thumbs Up,love ,victory ,Thumbs Down  by using single hand or two hands for various backgrounds ", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2, cv2.LINE_AA )

    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Check for key press events
    key = cv2.waitKey(1) & 0xFF

    # Right arrow key - Next Filter
    if key == ord('n'):
        filterIndex = (filterIndex + 1) % len(filters)

    # Left arrow key - Previous Filter
    elif key == ord('p'):
        filterIndex = (filterIndex - 1) % len(filters)

    for (x, y, w, h) in faces:
        # Grab the face
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape  # A Copy for future reference
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation=cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predicting the keypoints using the model
        keypoints = my_model.predict(face_resized)

        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the original image
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation=cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # Pair them together
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        # Add FILTER to the frame
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0] - points[9][0]) * 1.1)
        sunglass_height = int((points[10][1] - points[8][1]) / 1.1)
        sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_CUBIC)

        transparent_region = sunglass_resized[:, :, :3] != 0
        y_start = int(points[9][1])
        x_start = int(points[9][0])
        y_end = min(y_start + sunglass_height, face_resized_color.shape[0])
        x_end = min(x_start + sunglass_width, face_resized_color.shape[1])
        face_resized_color[y_start:y_end, x_start:x_end, :][transparent_region[:y_end-y_start, :x_end-x_start]] = sunglass_resized[:y_end-y_start, :x_end-x_start, :3][transparent_region[:y_end-y_start, :x_end-x_start]]

        # Resize the face_resized_color image back to its original shape
        frame[y:y + h, x:x + w] = cv2.resize(face_resized_color, original_shape, interpolation=cv2.INTER_CUBIC)

        for keypoint in points:
            # Convert floating-point coordinates to integers
            x = int(keypoint[0])
            y = int(keypoint[1])
            # Draw circle at the keypoint coordinates
            cv2.circle(face_resized_color2, (x, y), 1, (0, 255, 0), 1)

        # Show the frame2
        '''frame2 = np.copy( frame )
        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)
        cv2.imshow( "Facial Keypoints", frame2 )'''
        # Show the frame
        cv2.imshow("Realtime Face Detection with Filters", frame)

    # If the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
