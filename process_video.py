import cv2
import numpy as np
from collections import deque

# =========================
# CONFIG
# =========================
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"

MIN_AREA = 300
MAX_DIST = 260
CURVE_BEND = 0.45

LINE_THICKNESS = 1          
ALPHA = 0.42                # stronger visibility
POINT_MEMORY = 12           # frames to persist

# =========================
# BEZIER CURVE
# =========================
def bezier(p0, p1, p2, steps=30):
    return [
        (
            int((1 - t)**2 * p0[0] + 2*(1 - t)*t*p1[0] + t**2*p2[0]),
            int((1 - t)**2 * p0[1] + 2*(1 - t)*t*p1[1] + t**2*p2[1])
        )
        for t in np.linspace(0, 1, steps)
    ]

# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(INPUT_VIDEO)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# =========================
# BACKGROUND SUBTRACTOR
# =========================
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=600,
    varThreshold=30,
    detectShadows=False
)

# =========================
# WARM-UP
# =========================
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    fgbg.apply(frame)

# =========================
# TEMPORAL MEMORY
# =========================
point_history = deque(maxlen=POINT_MEMORY)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame, learningRate=0.003)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

    _, fgmask = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, w_, h_ = cv2.boundingRect(c)
        points.append((x + w_ // 2, y + h_ // 2))

    # Add current frame points to memory
    if points:
        point_history.append(points)

    overlay = frame.copy()

    # =========================
    # DRAW CURVES FROM MEMORY
    # =========================
    for past_points in point_history:
        for i in range(len(past_points)):
            for j in range(i + 1, len(past_points)):
                p0 = past_points[i]
                p2 = past_points[j]

                dist = np.linalg.norm(np.array(p0) - np.array(p2))
                if dist > MAX_DIST:
                    continue

                mid = (
                    int((p0[0] + p2[0]) / 2),
                    int((p0[1] + p2[1]) / 2 - dist * CURVE_BEND)
                )

                curve = bezier(p0, mid, p2)

                # Glow pass
                for k in range(len(curve) - 1):
                    cv2.line(
                        overlay,
                        curve[k],
                        curve[k + 1],
                        (255, 255, 255),
                        LINE_THICKNESS + 1,
                        cv2.LINE_AA
                    )

                # Core line
                for k in range(len(curve) - 1):
                    cv2.line(
                        overlay,
                        curve[k],
                        curve[k + 1],
                        (255, 255, 255),
                        LINE_THICKNESS,
                        cv2.LINE_AA
                    )

    # =========================
    # BLEND & WRITE
    # =========================
    frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
    writer.write(frame)

cap.release()
writer.release()

print("✅ Output saved as output.mp4")
