import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "No Hand Detected"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark

            fingers = [
                lm[8].y < lm[6].y,
                lm[12].y < lm[10].y,
                lm[16].y < lm[14].y,
                lm[20].y < lm[18].y
            ]

            thumb = lm[4].x < lm[3].x

            count = sum(fingers) + thumb

            if count == 5:
                gesture = "OPEN PALM"
            elif count == 0:
                gesture = "FIST"
            elif fingers[0] and not any(fingers[1:]) and not thumb:
                gesture = "ONE FINGER"
            elif fingers[0] and fingers[1] and not any(fingers[2:]):
                gesture = "TWO FINGERS"
            else:
                gesture = "OTHER GESTURE"

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame, gesture, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()