import cv2
import numpy as np
import streamlit as st

st.title("Green Screen Color Range Tuner")

# sliders
lh = st.slider("Lower Hue", 0, 179, 35)
ls = st.slider("Lower Sat", 0, 255, 60)
lv = st.slider("Lower Val", 0, 255, 60)
uh = st.slider("Upper Hue", 0, 179, 85)
us = st.slider("Upper Sat", 0, 255, 255)
uv = st.slider("Upper Val", 0, 255, 255)

# capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower, upper)
    result = frame.copy()
    result[mask > 0] = (255, 0, 0)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Green Mask", width='stretch')
else:
    st.warning("No webcam frame captured.")
