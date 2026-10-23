import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button
import math
import time
import ctypes  # For hardware-level mouse look (works inside Minecraft)

# ─── SETTINGS ───────────────────────────────────────────────
COOLDOWN = 0.4
# Mouse look sensitivity — tune this up/down to taste
MOUSE_SENSITIVITY = 1.8
# Smoothing of the fingertip position (0.1 = heavy smoothing, 1.0 = raw/snappy)
MOUSE_SMOOTHING = 0.35
# Soft dead zone (camera pixels per frame). Lower = finer aiming, higher = less jitter.
DEAD_ZONE = 1.5
# A new left-hand gesture must be seen this many frames in a row before it takes effect.
GESTURE_CONFIRM_FRAMES = 3
# 0 = faster/lighter hand model, 1 = more accurate (MediaPipe default)
MODEL_COMPLEXITY = 1

# ── Finger detection (works with palm OR back of hand facing the camera) ──
# A finger counts as "up" when it is bent less than this many degrees at its joints.
# Raise it if straight fingers are missed, lower it if half-curled fingers count as up.
FINGER_BEND_MAX = 60
# The thumb counts as "out" when its tip is at least this far from the base of the
# index finger, measured in palm lengths. Raise it if a tucked thumb counts as up,
# lower it if a thumbs-up isn't detected.
THUMB_OUT_RATIO = 0.6
# Use MediaPipe's depth estimate too, so fingers pointing toward/away from the
# camera are still measured correctly. Set False if detection gets jittery.
USE_DEPTH = True

# How to decide which hand is which:
#   "position" = left side of the (mirrored) screen is your left hand. Reliable even when
#                a hand is turned around, which can fool MediaPipe's left/right label.
#   "label"    = trust MediaPipe's left/right label (original behaviour).
HAND_ASSIGNMENT = "position"

# Show the detected finger states on screen (handy for tuning the values above)
SHOW_FINGER_DEBUG = True

# ─── CONTROLLERS ────────────────────────────────────────────
keyboard = Controller()
mouse = MouseController()


def win32_mouse_move_raw(dx, dy):
    """Relative mouse move via the Win32 API (bypasses Minecraft's cursor lock)."""
    ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)


# ─── MEDIAPIPE ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame -> less input lag

# Landmark indices for each finger: (MCP, PIP, DIP, TIP)
FINGER_JOINTS = [
    (5, 6, 7, 8),      # index
    (9, 10, 11, 12),   # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
]

# ─── STATE ──────────────────────────────────────────────────
held_keys = set()
held_buttons = set()

# Gesture debouncing
stable_gesture = "NONE"
candidate_gesture = "NONE"
candidate_count = 0

# Mouse look state
smooth_x = None
smooth_y = None
remainder_x = 0.0
remainder_y = 0.0

last_triggered = {"inventory": 0, "scroll": 0, "build": 0}

# Seconds between block placements while the BUILD gesture (pinky up) is held.
# Minecraft's own repeat when holding right-click is 0.2 s; higher = slower placing.
# Set to 0 to go back to holding right-click (needed for eating, bows, shields).
BUILD_REPEAT_DELAY = 0.6

# Right-hand hotbar scrolling (debounced the same way as the left hand)
SCROLL_COOLDOWN = 0.25   # minimum seconds between two hotbar scrolls
right_stable = "NONE"
right_candidate = "NONE"
right_count = 0

# FPS counter
prev_frame_time = time.time()
fps = 0.0

# Gesture -> held key
KEY_FOR_GESTURE = {
    "FORWARD": 'w',
    "BACKWARD": 's',
    "LEFT": 'a',
    "RIGHT": 'd',
    "SHIFT": Key.shift,
    "JUMP": Key.space,
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


def hold_button(button, should_hold):
    if should_hold and button not in held_buttons:
        mouse.press(button)
        held_buttons.add(button)
    elif not should_hold and button in held_buttons:
        mouse.release(button)
        held_buttons.discard(button)


def release_all_buttons():
    for button in list(held_buttons):
        mouse.release(button)
    held_buttons.clear()


def cooldown_ready(action, cooldown=COOLDOWN):
    return time.time() - last_triggered[action] > cooldown


def trigger(action):
    last_triggered[action] = time.time()


# ─── ORIENTATION-INDEPENDENT FINGER DETECTION ───────────────
def _vec(a, b):
    return tuple(bb - aa for aa, bb in zip(a, b))


def _dist(a, b):
    return math.sqrt(sum((bb - aa) ** 2 for aa, bb in zip(a, b)))


def _angle_between(v1, v2):
    """Angle in degrees between two vectors (0 = same direction)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _pt(lm_list, i):
    """Point used for finger-shape measurements (3D if USE_DEPTH)."""
    return lm_list[i] if USE_DEPTH else lm_list[i][:2]


def finger_is_extended(lm_list, mcp, pip, dip, tip):
    """
    A finger is 'up' when it is straight: the bend at the middle knuckle plus the
    bend at the last knuckle is small. This only depends on the shape of the finger,
    not on which way the hand is facing or pointing, so palm-facing, back-facing and
    sideways hands all give the same result.
    """
    p_mcp, p_pip, p_dip, p_tip = (_pt(lm_list, i) for i in (mcp, pip, dip, tip))
    bend_pip = _angle_between(_vec(p_mcp, p_pip), _vec(p_pip, p_dip))
    bend_dip = _angle_between(_vec(p_pip, p_dip), _vec(p_dip, p_tip))
    straight = (bend_pip + bend_dip) < FINGER_BEND_MAX * 1.5 and bend_pip < FINGER_BEND_MAX

    # Extra check: a straight finger's tip is farther from the wrist than its middle knuckle
    wrist = _pt(lm_list, 0)
    reaches_out = _dist(wrist, p_tip) > _dist(wrist, p_pip)
    return straight and reaches_out


def thumb_is_extended(lm_list):
    """
    The thumb is 'out' when its tip is far from the base of the index finger
    (relative to palm size) and it is pointing away from the palm.
    No left/right x-comparison, so it doesn't flip when the hand turns around.
    """
    palm_size = _dist(_pt(lm_list, 0), _pt(lm_list, 9))
    if palm_size == 0:
        return False
    tip, ip, index_mcp = _pt(lm_list, 4), _pt(lm_list, 3), _pt(lm_list, 5)
    far_from_index = _dist(tip, index_mcp) / palm_size > THUMB_OUT_RATIO
    pointing_away = _dist(tip, index_mcp) > _dist(ip, index_mcp)
    return far_from_index and pointing_away


def get_fingers(lm_list):
    """Returns [thumb, index, middle, ring, pinky] as 1 (up) / 0 (down)."""
    fingers = [1 if thumb_is_extended(lm_list) else 0]
    for joints in FINGER_JOINTS:
        fingers.append(1 if finger_is_extended(lm_list, *joints) else 0)
    return fingers


def get_pointing_state(lm_list, fingers):
    """
    Evaluates pointing directions and finger counts for the Left hand.
    Direction uses the on-screen (2D) direction of the finger, which is the same
    whether the palm or the back of the hand faces the camera.
    Returns a tuple: (finger_mode, direction)
    """
    thumb, index, middle, ring, pinky = fingers

    # Movement modes ignore the thumb and pinky here; those are checked as
    # jump/sneak modifiers in identify_gesture().
    if thumb and not index and not middle and not ring and not pinky:
        mode = "THUMB"
    elif index and not middle and not ring:
        mode = "INDEX"
    elif index and middle and not ring:
        mode = "INDEX_MIDDLE"
    else:
        return None, None

    if mode == "THUMB":
        tip_x, tip_y = lm_list[4][:2]
        base_x, base_y = lm_list[2][:2]
    else:
        tip_x, tip_y = lm_list[8][:2]
        base_x, base_y = lm_list[5][:2]

    dx = tip_x - base_x
    dy = tip_y - base_y

    if math.hypot(dx, dy) < 20:
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


def get_movement(mode, direction):
    if direction == "UP" and mode == "INDEX":          return "FORWARD"
    if direction == "UP" and mode == "INDEX_MIDDLE":   return "BACKWARD"
    if direction == "RIGHT":
        if mode == "INDEX":                            return "RIGHT"
        if mode == "INDEX_MIDDLE":                     return "LEFT"
    return None


def identify_gesture(lm_list):
    """
    Returns (gesture_name, finger_list).
    While moving, modifiers are added with '+', e.g. "FORWARD+JUMP" or "LEFT+SHIFT".
    """
    if lm_list is None:
        return "NONE", None

    f = get_fingers(lm_list)
    thumb, pinky = f[0], f[4]
    mode, direction = get_pointing_state(lm_list, f)

    # Standing still: thumbs up = jump (unchanged)
    if mode == "THUMB" and direction == "UP":          return "JUMP", f

    movement = get_movement(mode, direction)
    if movement is not None:
        # While moving:
        #   thumb up            -> also jump
        #   thumb + pinky up    -> also sneak
        #   pinky up, no thumb  -> not a move (index+pinky stays BREAK)
        if thumb and pinky:     return movement + "+SHIFT", f
        if thumb:               return movement + "+JUMP", f
        if not pinky:           return movement, f

    four_fingers = f[1:]  # Index, Middle, Ring, Pinky (ignores thumb)

    if f == REST:               return "REST", f
    if f == SHIFT_G:            return "SHIFT", f
    if f == INVENTORY:          return "INVENTORY", f

    if four_fingers == [1, 0, 0, 1]:  return "BREAK", f  # Index and Pinky are UP
    if four_fingers == [0, 0, 0, 1]:  return "BUILD", f  # Only Pinky is UP

    return "UNKNOWN", f


def identify_right_gesture(lm_list):
    """
    Right hand hotbar gestures. Only the finger alone counts, so the thumb sticking
    out naturally while you aim with your index finger doesn't scroll.
      Pinky only -> SCROLL_RIGHT
      Thumb only -> SCROLL_LEFT
    """
    if lm_list is None:
        return "NONE"
    f = get_fingers(lm_list)
    if f == [0, 0, 0, 0, 1]:
        return "SCROLL_RIGHT"
    if f == [1, 0, 0, 0, 0]:
        return "SCROLL_LEFT"
    return "OTHER"


def split_hands(results, w, h):
    """
    Returns (left_lm_list, right_lm_list, raw_landmarks).
    Each landmark is (x, y, z) in camera pixels; z is MediaPipe's depth estimate
    scaled to the same units as x.
    """
    found = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            label = handedness.classification[0].label
            pts = [(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks.landmark]
            found.append([label, pts, hand_landmarks])

    if HAND_ASSIGNMENT == "position" or (
        len(found) == 2 and found[0][0] == found[1][0]
    ):
        # Mirrored image: your left hand is on the left side of the screen
        if len(found) == 2:
            found.sort(key=lambda item: item[1][0][0])  # sort by wrist x
            found[0][0], found[1][0] = "Left", "Right"
        elif len(found) == 1:
            found[0][0] = "Left" if found[0][1][0][0] < w / 2 else "Right"

    left, right = None, None
    for label, pts, _ in found:
        if label == "Left":
            left = pts
        else:
            right = pts
    return left, right, [f[2] for f in found]


# HUD text (OpenCV can't draw emoji, so plain text only)
GESTURE_DISPLAY = [
    ("FORWARD", "Index UP"),
    ("BACKWARD", "Index+Middle UP"),
    ("LEFT", "Index+Middle RIGHT"),
    ("RIGHT", "Index RIGHT"),
    ("JUMP", "Thumbs UP / moving + thumb"),
    ("SHIFT", "Thumb+Pinky / moving + thumb+pinky"),
    ("BREAK", "Index+Pinky"),
    ("BUILD", "Pinky only"),
    ("INVENTORY", "Open hand"),
    ("REST", "Fist"),
]


# ─── MAIN LOOP ──────────────────────────────────────────────
try:
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # lets MediaPipe skip a copy
        results = hands.process(img_rgb)

        left_lm_list, right_lm_list, raw_landmarks = split_hands(results, w, h)
        for hand_landmarks in raw_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ── MOUSE LOOK (Right Hand Index Tracking) ─────────────
        right_hand_tracking = (
            right_lm_list is not None
            and finger_is_extended(right_lm_list, *FINGER_JOINTS[0])
        )

        if right_hand_tracking:
            raw_x, raw_y = right_lm_list[8][:2]

            if smooth_x is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                new_x = smooth_x + (raw_x - smooth_x) * MOUSE_SMOOTHING
                new_y = smooth_y + (raw_y - smooth_y) * MOUSE_SMOOTHING
                dx = new_x - smooth_x
                dy = new_y - smooth_y
                smooth_x, smooth_y = new_x, new_y

                # Soft dead zone
                dx = math.copysign(max(0.0, abs(dx) - DEAD_ZONE), dx)
                dy = math.copysign(max(0.0, abs(dy) - DEAD_ZONE), dy)

                # Accumulate fractions so tiny, slow movements still add up
                remainder_x += dx * MOUSE_SENSITIVITY
                remainder_y += dy * MOUSE_SENSITIVITY
                move_x, move_y = int(remainder_x), int(remainder_y)
                remainder_x -= move_x
                remainder_y -= move_y

                if move_x or move_y:
                    win32_mouse_move_raw(move_x, move_y)
        else:
            smooth_x = smooth_y = None
            remainder_x = remainder_y = 0.0

        # ── RIGHT HAND HOTBAR SCROLL (one slot per raise) ──────
        right_raw = identify_right_gesture(right_lm_list)
        if right_raw == right_candidate:
            right_count += 1
        else:
            right_candidate = right_raw
            right_count = 1

        prev_right = right_stable
        if right_count >= GESTURE_CONFIRM_FRAMES:
            right_stable = right_candidate

        if right_stable != prev_right and cooldown_ready("scroll", SCROLL_COOLDOWN):
            if right_stable == "SCROLL_RIGHT":
                mouse.scroll(0, -1)   # scroll down = next slot (to the right)
                trigger("scroll")
            elif right_stable == "SCROLL_LEFT":
                mouse.scroll(0, 1)    # scroll up = previous slot (to the left)
                trigger("scroll")

        # ── LEFT HAND GESTURES (debounced) ─────────────────────
        raw_gesture, left_fingers = identify_gesture(left_lm_list)

        if raw_gesture == candidate_gesture:
            candidate_count += 1
        else:
            candidate_gesture = raw_gesture
            candidate_count = 1

        prev_gesture = stable_gesture
        if candidate_count >= GESTURE_CONFIRM_FRAMES:
            stable_gesture = candidate_gesture
        current_gesture = stable_gesture
        gesture_just_changed = current_gesture != prev_gesture

        # Movement + action keys ("FORWARD+JUMP" holds both W and Space)
        active_actions = set(current_gesture.split("+"))
        for action, key in KEY_FOR_GESTURE.items():
            if action in active_actions:
                press_key(key)
            else:
                release_key(key)

        # Break (Left Click) / Build (Right Click)
        hold_button(Button.left, current_gesture == "BREAK")
        if BUILD_REPEAT_DELAY > 0:
            # Place one block right away, then one more every BUILD_REPEAT_DELAY seconds
            if current_gesture == "BUILD" and (
                gesture_just_changed or cooldown_ready("build", BUILD_REPEAT_DELAY)
            ):
                mouse.click(Button.right)
                trigger("build")
        else:
            hold_button(Button.right, current_gesture == "BUILD")

        # Inventory
        if current_gesture == "INVENTORY" and gesture_just_changed and cooldown_ready("inventory"):
            keyboard.press('e')
            keyboard.release('e')
            trigger("inventory")

        if current_gesture in ("REST", "NONE", "UNKNOWN"):
            release_all()

        # ── DRAW DIRECTION ARROW (Left hand movement indicator) ──
        if left_lm_list is not None:
            tip = tuple(int(v) for v in left_lm_list[8][:2])
            base = tuple(int(v) for v in left_lm_list[5][:2])
            cv2.arrowedLine(img, base, tip, (0, 255, 255), 3)

        # ── HUD ────────────────────────────────────────────────
        for i, (action, gesture) in enumerate(GESTURE_DISPLAY):
            color = (0, 255, 0) if action in active_actions else (200, 200, 200)
            cv2.putText(img, f"{action}: {gesture}", (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        right_status = "TRACKING (Index Up)" if right_hand_tracking else "PAUSED"
        status_color = (0, 255, 0) if right_hand_tracking else (0, 0, 255)
        cv2.putText(img, f"Right Hand: {right_status}", (w - 280, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        hotbar_text = {"SCROLL_RIGHT": "Hotbar: >> (Pinky)",
                       "SCROLL_LEFT": "Hotbar: << (Thumb)"}.get(right_stable, "Hotbar: -")
        hotbar_color = (0, 255, 0) if right_stable in ("SCROLL_RIGHT", "SCROLL_LEFT") else (200, 200, 200)
        cv2.putText(img, hotbar_text, (w - 280, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hotbar_color, 1)
        cv2.putText(img, f"Gesture: {current_gesture}", (w - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_time, 1e-6))
        prev_frame_time = now
        cv2.putText(img, f"FPS: {fps:.0f}", (w - 280, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if SHOW_FINGER_DEBUG and left_fingers is not None:
            names = ["T", "I", "M", "R", "P"]
            text = "Left: " + " ".join(f"{n}{v}" for n, v in zip(names, left_fingers))
            cv2.putText(img, text, (w - 280, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cv2.imshow("Minecraft Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Runs even if the script crashes, so you never get stuck holding W or clicking
    release_all()
    release_all_buttons()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()