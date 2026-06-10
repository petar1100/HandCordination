import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button
import pyautogui
import math
import time
import ctypes  # For hardware-level DirectInput mouse look bypass

# ─── SETTINGS ───────────────────────────────────────────────
pyautogui.FAILSAFE = False
COOLDOWN = 0.4
# Mouse look sensitivity — tune this up/down to taste
MOUSE_SENSITIVITY = 1.8
# Smoothing factor (0.1 = heavy smoothing, 1.0 = raw/snappy)
MOUSE_SMOOTHING = 0.35
# Dead zone: ignore small finger movements to reduce jitter (pixels in cam space)
DEAD_ZONE = 8
screen_w, screen_h = pyautogui.size()

# ─── CONTROLLERS ────────────────────────────────────────────
keyboard = Controller()
mouse = MouseController()


# Hardware-level mouse injector function to bypass Minecraft engine lock
def win32_mouse_move_raw(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)


# ─── MEDIAPIPE ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

# ─── STATE ──────────────────────────────────────────────────
left_fingers = None
right_lm_list = None
held_keys = set()
prev_left_gesture = None
build_active = False
break_active = False

# Mouse delta smoothing variables
smooth_dx = 0
smooth_dy = 0

# Previous right-hand index fingertip position (in camera pixels)
prev_raw_x = None
prev_raw_y = None

last_triggered = {
    "inventory": 0
}


# ─── HELPERS ────────────────────────────────────────────────
def press_key(key):
    if key not in held_keys:
        keyboard.press(key)
        held_keys.add(key)


def release_key(key):
    if key in held_keys:
        keyboard.release(key)
        held_keys.discard(key)


def release_all():
    for key in list(held_keys):
        keyboard.release(key)
    held_keys.clear()


def cooldown_ready(action):
    return time.time() - last_triggered[action] > COOLDOWN


def trigger(action):
    last_triggered[action] = time.time()


def get_fingers(lm_list, label):
    fingers = []
    if label == "Right":
        fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
    else:
        fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)
    for i in range(1, 5):
        fingers.append(1 if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1] else 0)
    return fingers


def get_pointing_state(lm_list):
    """
    Evaluates pointing directions and finger counts for the Left hand.
    Returns a tuple: (finger_mode, direction)
    """
    fingers = get_fingers(lm_list, "Left")
    thumb, index, middle, ring, pinky = fingers[0], fingers[1], fingers[2], fingers[3], fingers[4]

    if thumb and not index and not middle and not ring and not pinky:
        mode = "THUMB"
    elif index and not middle and not ring and not pinky:
        mode = "INDEX"
    elif index and middle and not ring and not pinky:
        mode = "INDEX_MIDDLE"
    else:
        return None, None

    if mode == "THUMB":
        tip_x, tip_y = lm_list[4][0], lm_list[4][1]
        base_x, base_y = lm_list[2][0], lm_list[2][1]
    else:
        tip_x, tip_y = lm_list[8][0], lm_list[8][1]
        base_x, base_y = lm_list[5][0], lm_list[5][1]

    dx = tip_x - base_x
    dy = tip_y - base_y

    length = math.sqrt(dx ** 2 + dy ** 2)
    if length < 20:
        return None, None

    angle = math.degrees(math.atan2(dy, dx))

    if -135 <= angle < -45:
        direction = "UP"
    elif 45 <= angle < 135:
        direction = "DOWN"
    elif angle >= 135 or angle < -135:
        direction = "LEFT"
    else:
        direction = "RIGHT"

    return mode, direction


# ─── GESTURE DEFINITIONS ────────────────────────────────────
REST = [0, 0, 0, 0, 0]
SHIFT_G = [1, 0, 0, 0, 1]  # Thumb + Pinky up handles Shift (Sneak)
INVENTORY = [1, 1, 1, 1, 1]


def identify_gesture(lm_list, label):
    if lm_list is None:
        return "NONE"

    mode, direction = get_pointing_state(lm_list)

    if mode == "THUMB" and direction == "UP":          return "JUMP"
    if direction == "UP" and mode == "INDEX":          return "FORWARD"
    if direction == "UP" and mode == "INDEX_MIDDLE":   return "BACKWARD"
    if direction == "RIGHT":
        if mode == "INDEX":                            return "RIGHT"
        if mode == "INDEX_MIDDLE":                     return "LEFT"

    f = get_fingers(lm_list, label)

    # Extract just the 4 main fingers (Index, Middle, Ring, Pinky) to bypass thumb bugs
    four_fingers = f[1:]

    if f == REST:               return "REST"
    if f == SHIFT_G:            return "SHIFT"
    if f == INVENTORY:          return "INVENTORY"

    # Relaxed tracking logic: registers BREAK regardless of your thumb's position
    if four_fingers == [1, 0, 0, 1]:  return "BREAK"  # Index and Pinky are UP
    if four_fingers == [0, 0, 0, 1]:  return "BUILD"  # Only Pinky is UP

    return "UNKNOWN"


# ─── MAIN LOOP ──────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    left_lm_list = None
    right_lm_list = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
        ):
            label = handedness.classification[0].label
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if label == "Left":
                left_lm_list = lm_list
            else:
                right_lm_list = lm_list

    # ── MOUSE LOOK (Right Hand Index Tracking) ─────────────────
    right_hand_tracking = False

    if right_lm_list is not None:
        right_f = get_fingers(right_lm_list, "Right")
        if right_f[1] == 1:
            right_hand_tracking = True

    if right_hand_tracking:
        raw_x = right_lm_list[8][0]
        raw_y = right_lm_list[8][1]

        if prev_raw_x is not None and prev_raw_y is not None:
            target_dx = raw_x - prev_raw_x
            target_dy = raw_y - prev_raw_y

            if abs(target_dx) < DEAD_ZONE: target_dx = 0
            if abs(target_dy) < DEAD_ZONE: target_dy = 0

            smooth_dx += (target_dx - smooth_dx) * MOUSE_SMOOTHING
            smooth_dy += (target_dy - smooth_dy) * MOUSE_SMOOTHING

            if abs(smooth_dx) > 0.1 or abs(smooth_dy) > 0.1:
                win32_mouse_move_raw(smooth_dx * MOUSE_SENSITIVITY, smooth_dy * MOUSE_SENSITIVITY)

        prev_raw_x = raw_x
        prev_raw_y = raw_y
    else:
        prev_raw_x = None
        prev_raw_y = None
        smooth_dx = 0
        smooth_dy = 0

    # ── LEFT HAND GESTURES ───────────────────────────────────
    current_gesture = identify_gesture(left_lm_list, "Left")
    gesture_just_changed = current_gesture != prev_left_gesture

    # Movement Keys
    if current_gesture == "FORWARD":
        press_key('w')
    else:
        release_key('w')

    if current_gesture == "BACKWARD":
        press_key('s')
    else:
        release_key('s')

    if current_gesture == "LEFT":
        press_key('a')
    else:
        release_key('a')

    if current_gesture == "RIGHT":
        press_key('d')
    else:
        release_key('d')

    # Action Keys
    if current_gesture == "SHIFT":
        press_key(Key.shift)
    else:
        release_key(Key.shift)

    if current_gesture == "JUMP":
        press_key(Key.space)
    else:
        release_key(Key.space)

    # Break Actions (Left Click)
    if current_gesture == "BREAK" and gesture_just_changed:
        if not break_active:
            mouse.press(Button.left)
            break_active = True
    elif current_gesture != "BREAK" and break_active:
        mouse.release(Button.left)
        break_active = False

    # Build Actions (Right Click)
    if current_gesture == "BUILD" and gesture_just_changed:
        if not build_active:
            mouse.press(Button.right)
            build_active = True
    elif current_gesture != "BUILD" and build_active:
        mouse.release(Button.right)
        build_active = False

    # Menu Triggers
    if current_gesture == "INVENTORY" and gesture_just_changed:
        if cooldown_ready("inventory"):
            keyboard.press('e')
            keyboard.release('e')
            trigger("inventory")

    if current_gesture in ("REST", "NONE", "UNKNOWN"):
        release_all()

    prev_left_gesture = current_gesture

    # ── DRAW DIRECTION ARROW (Left hand movement indicator) ──
    if left_lm_list is not None:
        tip_x, tip_y = left_lm_list[8][0], left_lm_list[8][1]
        base_x, base_y = left_lm_list[5][0], left_lm_list[5][1]
        cv2.arrowedLine(img, (base_x, base_y), (tip_x, tip_y), (0, 255, 255), 3)

    # ── HUD ──────────────────────────────────────────────────
    gesture_display = [
        ("FORWARD", "☝️  Index UP"),
        ("BACKWARD", "✌️  Index+Middle UP"),
        ("LEFT", "👈 Index+Middle RIGHT"),
        ("RIGHT", "👉 Index RIGHT"),
        ("JUMP", "👍 Thumbs UP"),
        ("SHIFT", "🤙 Thumb+Pinky"),
        ("BREAK", "🤘 Index+Pinky"),
        ("BUILD", "🖕 Pinky only"),
        ("INVENTORY", "🖐️  Open hand"),
        ("REST", "👊 Fist"),
    ]

    for i, (action, gesture) in enumerate(gesture_display):
        color = (0, 255, 0) if action == current_gesture else (200, 200, 200)
        cv2.putText(img, f"{action}: {gesture}", (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Display Right Hand Look Status on UI
    right_status = "TRACKING (Index Up)" if right_hand_tracking else "PAUSED"
    status_color = (0, 255, 0) if right_hand_tracking else (0, 0, 255)
    cv2.putText(img, f"Right Hand: {right_status}", (w - 280, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    cv2.putText(img, f"Gesture: {current_gesture}", (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Minecraft Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ── CLEANUP ──────────────────────────────────────────────────
release_all()
mouse.release(Button.left)
mouse.release(Button.right)
cap.release()
cv2.destroyAllWindows()