import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# MediaPipe setup
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# White virtual notepad
# -----------------------------

notepad = np.ones((480, 640, 3), dtype=np.uint8) * 255

previous_point = None

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access camera.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # -----------------------------
    # Detect hand
    # -----------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # -----------------------------
        # Get index fingertip
        # -----------------------------

        fingertip = hand.landmark[8]

        h, w, _ = frame.shape

        # Position on webcam
        finger_x = int(fingertip.x * w)
        finger_y = int(fingertip.y * h)

        # Show fingertip
        cv2.circle(
            frame,
            (finger_x, finger_y),
            12,
            (255, 255, 255),
            -1
        )

        # -----------------------------
        # Convert to notepad position
        # -----------------------------

        x = int(fingertip.x * 640)
        y = int(fingertip.y * 480)

        # Keep inside notepad
        x = max(0, min(639, x))
        y = max(0, min(479, y))

        current_point = (x, y)

        # -----------------------------
        # DRAW DIRECTLY ON NOTEPAD
        # -----------------------------

        if previous_point is not None:

            cv2.line(
                notepad,
                previous_point,
                current_point,
                (0, 0, 0),
                6
            )

        # Draw a small cursor
        cv2.circle(
            notepad,
            current_point,
            8,
            (0, 0, 0),
            -1
        )

        previous_point = current_point

    else:

        # Hand disappeared
        previous_point = None

    # -----------------------------
    # Labels
    # -----------------------------

    cv2.putText(
        frame,
        "WEBCAM",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        notepad,
        "VIRTUAL NOTEPAD",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    cv2.putText(
        notepad,
        "C = Clear    Q = Quit",
        (20, 465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )

    # -----------------------------
    # Show both
    # -----------------------------

    combined = np.hstack((frame, notepad))

    cv2.imshow(
        "MagicHand - Air Drawing",
        combined
    )

    # -----------------------------
    # Keyboard
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        notepad = np.ones(
            (480, 640, 3),
            dtype=np.uint8
        ) * 255
        previous_point = None

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()