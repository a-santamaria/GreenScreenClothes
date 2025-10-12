import cv2
import numpy as np

def nothing(x): pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

# Hue 0–179, Saturation/Value 0–255
cv2.createTrackbar("LH", "Trackbars", 35, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 60, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 60, 255, nothing)
cv2.createTrackbar("UH", "Trackbars", 85, 179, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower, upper)

    result = frame.copy()
    result[mask > 0] = (255, 0, 0)

    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()