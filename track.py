import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time

keyboard = Controller()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append((cx, cy))

            fingers = []

            if lm_list:

                # Thumb
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers
                for id in range(1, 5):

                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id]-2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Count fingers
                total_fingers = fingers.count(1)

                # 🎮 Gesture → Minecraft control

                # ✊ Fist → Forward
                if total_fingers == 0:
                    time.sleep(0.1)
                    keyboard.press('w')
                else:
                    keyboard.release('w')

                # ✋ Open hand → Back
                if total_fingers == 5:
                    time.sleep(0.1)
                    keyboard.press('s')
                else:
                    keyboard.release('s')

                # ✌️ Two fingers → Jump
                if total_fingers == 2:
                    time.sleep(0.1)
                    keyboard.press(Key.space)
                else:
                    keyboard.release(Key.space)

                # 👍 Thumb → Sneak
                if fingers == [1,0,0,0,0]:
                    time.sleep(0.1)
                    keyboard.press(Key.shift)
                else:
                    keyboard.release(Key.shift)

    cv2.imshow("Minecraft Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()