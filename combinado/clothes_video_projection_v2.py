import cv2
import numpy as np
import sys
import time
import os
import glob


class VideoLooper:
    """Cycles through video files in a 'videos' folder and provides frames sized to the webcam feed."""

    SUPPORTED = ("*.mp4", "*.mov", "*.avi", "*.mkv")

    def __init__(self, video_dir, frame_size):
        self.video_dir = video_dir
        self.w, self.h = frame_size
        self.paths = []
        self.index = 0
        self.cap = None
        self.current_name = None
        os.makedirs(self.video_dir, exist_ok=True)
        self.reload()

    def _release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _collect(self):
        paths = []
        for pattern in self.SUPPORTED:
            paths.extend(sorted(glob.glob(os.path.join(self.video_dir, pattern))))
        self.paths = paths
        if not self.paths:
            print(f"No videos found in {self.video_dir}.")

    def _open_current(self):
        self._release()
        if not self.paths:
            self.current_name = None
            return
        path = self.paths[self.index % len(self.paths)]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open {path}, skipping.")
            self.paths.pop(self.index)
            if self.paths:
                self.index %= len(self.paths)
                self._open_current()
            else:
                print("No playable videos remain.")
        else:
            self.cap = cap
            self.current_name = os.path.basename(path)
            print(f"Projecting: {self.current_name}")

    def reload(self):
        prev = self.current_name
        self._collect()
        if not self.paths:
            self._release()
            return
        if prev:
            for i, p in enumerate(self.paths):
                if os.path.basename(p) == prev:
                    self.index = i
                    break
        self.index %= len(self.paths)
        self._open_current()

    def set_frame_size(self, frame_size):
        self.w, self.h = frame_size

    def next_video(self):
        if not self.paths:
            print("No videos to advance.")
            return
        self.index = (self.index + 1) % len(self.paths)
        self._open_current()

    def next_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            # loop to next video
            if self.paths:
                self.index = (self.index + 1) % len(self.paths)
            self._open_current()
            if self.cap is None:
                return None
            return self.next_frame()
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return frame

    def current(self):
        return self.current_name

    def available(self):
        return [os.path.basename(p) for p in self.paths]

    def cleanup(self):
        self._release()


def handle_hsv_command(cmd, lower_hue, upper_hue, step):
    updated = False
    if cmd == 'h':
        if lower_hue > 0:
            lower_hue = max(0, lower_hue - step)
            if lower_hue >= upper_hue:
                upper_hue = min(179, lower_hue + 1)
            updated = True
    elif cmd == 'j':
        if lower_hue < upper_hue - 1:
            lower_hue = min(upper_hue - 1, lower_hue + step)
            updated = True
    elif cmd == 'k':
        if lower_hue < upper_hue - 1:
            upper_hue = max(lower_hue + 1, upper_hue - step)
            updated = True
    elif cmd == 'l':
        if upper_hue < 179:
            upper_hue = min(179, upper_hue + step)
            updated = True
    return lower_hue, upper_hue, updated


def build_overlay_lines(looper, lower_hue, upper_hue, show_original):
    current = looper.current() if looper else "None"
    total = len(looper.available()) if looper else 0
    position = ((looper.index % total) + 1) if looper and total else 0
    count_label = f"{position}/{total}" if total else "0/0"
    lines = [
        f"Hue: {lower_hue}-{upper_hue} (h/j/k/l)",
        f"Overlay: {'ON' if show_original else 'OFF'} (o)",
        f"Video: {current if current else 'None'} ({count_label})",
        "n: next video | r: reload list",
        "s: snapshot | ?: toggle controls",
        "q or ESC: quit",
    ]
    return lines


def draw_overlay(frame, lines):
    if not lines:
        return frame

    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding_x = 14
    padding_y = 12
    line_gap = 6

    # Compute panel bounds based on text sizes for consistent layout.
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_height = max(size[1] for size in text_sizes) if text_sizes else 0
    panel_width = max(size[0] for size in text_sizes) + padding_x * 2
    panel_height = len(lines) * (text_height + line_gap) - line_gap + padding_y * 2

    top_left = (10, 10)
    bottom_right = (10 + panel_width, 10 + panel_height)

    overlay = output.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)

    for idx, (line, size) in enumerate(zip(lines, text_sizes)):
        text_x = top_left[0] + padding_x
        text_y = top_left[1] + padding_y + idx * (text_height + line_gap) + size[1]
        cv2.putText(output, line, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return output


# HSV thresholds for mask
lower_hue = 30
upper_hue = 90
lower_sat = 50
upper_sat = 255
lower_val = 60
upper_val = 255
hue_step = 2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    sys.exit(1)

video_looper = None
show_original_video = True
show_instructions = True

try:
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if video_looper is None:
            video_looper = VideoLooper(video_dir=os.path.join(os.path.dirname(__file__), 'videos'), frame_size=(w, h))
        else:
            if video_looper.w != w or video_looper.h != h:
                video_looper.set_frame_size((w, h))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bounds = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
        upper_bounds = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

        vid_frame = video_looper.next_frame() if video_looper else None
        if vid_frame is None:
            vid_frame = np.zeros_like(frame)

        if show_original_video:
            result = frame.copy()
        else:
            result = np.zeros_like(frame)
        result[mask > 0] = vid_frame[mask > 0]

        overlay_lines = build_overlay_lines(video_looper, lower_hue, upper_hue, show_original_video) if show_instructions else []
        result = draw_overlay(result, overlay_lines)

        cv2.imshow('Result', result)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            print("Quitting (key press).")
            break
        elif key == ord('h'):
            lower_hue, upper_hue, _ = handle_hsv_command('h', lower_hue, upper_hue, hue_step)
        elif key == ord('j'):
            lower_hue, upper_hue, _ = handle_hsv_command('j', lower_hue, upper_hue, hue_step)
        elif key == ord('k'):
            lower_hue, upper_hue, _ = handle_hsv_command('k', lower_hue, upper_hue, hue_step)
        elif key == ord('l'):
            lower_hue, upper_hue, _ = handle_hsv_command('l', lower_hue, upper_hue, hue_step)
        elif key == ord('n'):
            if video_looper: video_looper.next_video()
        elif key == ord('o'):
            show_original_video = not show_original_video
        elif key == ord('r'):
            if video_looper: video_looper.reload()
        elif key == ord('s'):
            # Save current composite frame snapshot
            snapshot_name = f"projection_snapshot_{int(time.time())}.png"
            cv2.imwrite(snapshot_name, result)
            print(f"Saved snapshot {snapshot_name}")
        elif key == ord('?'):
            show_instructions = not show_instructions

finally:
    cap.release()
    cv2.destroyAllWindows()
    if video_looper: video_looper.cleanup()
    print("Cleaned up. Bye.")