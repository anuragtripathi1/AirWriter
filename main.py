import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
smooth_factor = 0.5
eraser_radius = 40

strokes = []
current_stroke = []
selected_stroke = None
last_pos = None

draw_color = (255, 0, 255)

# ⏱️ gesture timing
gesture_start = 0
current_gesture = None
GESTURE_DELAY = 0.4

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def fingers_status(lm):
    return [
        lm[8].y < lm[6].y,   # index
        lm[12].y < lm[10].y, # middle
        lm[16].y < lm[14].y, # ring
        lm[20].y < lm[18].y  # pinky
    ]

def detect_gesture(lm, index_tip, thumb_tip):
    fingers = fingers_status(lm)
    dist = distance(index_tip, thumb_tip)

    if fingers == [True, False, False, False]:
        return "DRAW"
    elif dist < 30:
        return "MOVE"
    elif all(fingers):
        return "ERASE"
    else:
        return "IDLE"

def get_nearest_stroke(point):
    for stroke in strokes:
        for p in stroke["points"]:
            if distance(point, p) < 40:
                return stroke
    return None

def redraw_canvas():
    global canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    for stroke in strokes:
        pts = stroke["points"]
        color = stroke["color"]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, 6)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 🎨 color buttons
    colors = [(255,0,255), (255,0,0), (0,255,0), (0,0,255)]
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (10+i*60,10), (60+i*60,60), col, -1)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            h, w, _ = frame.shape

            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            # 🎯 gesture detection with delay
            detected = detect_gesture(lm, index_tip, thumb_tip)

            if detected != current_gesture:
                current_gesture = detected
                gesture_start = time.time()

            if time.time() - gesture_start > GESTURE_DELAY:
                
                # 🎨 color select
                x, y = index_tip
                if y < 60:
                    for i in range(4):
                        if 10+i*60 < x < 60+i*60:
                            draw_color = colors[i]

                # ✍️ DRAW
                if current_gesture == "DRAW":
                    selected_stroke = None
                    last_pos = None

                    if prev_x == 0:
                        prev_x, prev_y = index_tip

                    smooth_x = int(prev_x + smooth_factor * (index_tip[0] - prev_x))
                    smooth_y = int(prev_y + smooth_factor * (index_tip[1] - prev_y))

                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, 6)
                    current_stroke.append((smooth_x, smooth_y))

                    prev_x, prev_y = smooth_x, smooth_y

                # 🤏 MOVE
                elif current_gesture == "MOVE":
                    prev_x, prev_y = 0, 0

                    if len(current_stroke) > 0:
                        strokes.append({"points": current_stroke, "color": draw_color})
                        current_stroke = []

                    if selected_stroke is None:
                        selected_stroke = get_nearest_stroke(index_tip)
                        last_pos = index_tip

                    if selected_stroke and last_pos:
                        dx = index_tip[0] - last_pos[0]
                        dy = index_tip[1] - last_pos[1]

                        for i in range(len(selected_stroke["points"])):
                            x, y = selected_stroke["points"][i]
                            selected_stroke["points"][i] = (x+dx, y+dy)

                        last_pos = index_tip
                        redraw_canvas()

                # 🧽 ERASE
                elif current_gesture == "ERASE":
                    prev_x, prev_y = 0, 0
                    current_stroke = []
                    selected_stroke = None
                    last_pos = None

                    new_strokes = []
                    for stroke in strokes:
                        new_points = [p for p in stroke["points"] if distance(p, index_tip) > eraser_radius]
                        if len(new_points) > 2:
                            stroke["points"] = new_points
                            new_strokes.append(stroke)

                    strokes = new_strokes
                    redraw_canvas()

                # 😴 IDLE
                else:
                    prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # clean overlay
    frame_with_canvas = frame.copy()
    mask = canvas.astype(bool)
    frame_with_canvas[mask] = canvas[mask]

    cv2.imshow("Air Writing Smart", frame_with_canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()