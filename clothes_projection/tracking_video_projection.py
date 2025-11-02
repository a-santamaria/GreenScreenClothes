import cv2
import numpy as np
import sys
import time
import os
import glob
import json


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
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    current_index = self.index
                    self._open_current()
                    if self.cap is None:
                        return None
                    if self.index != current_index:
                        self.index = current_index
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return frame

    def current(self):
        return self.current_name

    def available(self):
        return [os.path.basename(p) for p in self.paths]

    def cleanup(self):
        self._release()


class OpticalFlowTracker:
    """Maintains a garment mask by tracking feature points with Lucas-Kanade flow."""

    def __init__(self, max_corners=400, quality_level=0.01, min_distance=7, min_points=20):
        self.feature_params = dict(maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.min_points = min_points
        self.prev_gray = None
        self.prev_points = None
        self.last_mask = None
        self.active = False

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.last_mask = None
        self.active = False

    def initialize(self, frame_gray, hsv_mask):
        points = cv2.goodFeaturesToTrack(frame_gray, mask=hsv_mask, **self.feature_params)
        if points is None or len(points) < self.min_points:
            print("Tracking initialization failed: insufficient features in mask.")
            self.reset()
            return False
        self.prev_gray = frame_gray.copy()
        self.prev_points = points
        self.last_mask = self.build_mask_from_points(points.reshape(-1, 2), frame_gray.shape)
        self.active = True
        print(f"Tracking initialized with {len(points)} points.")
        return True

    def update(self, frame_gray, hsv_mask=None):
        if not self.active or self.prev_gray is None or self.prev_points is None:
            return None
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params)
        if next_points is None:
            print("Tracking update failed: optical flow returned no points.")
            self.reset()
            return None
        status = status.reshape(-1)
        good_new = next_points[status == 1]
        if good_new.size < self.min_points * 2:
            print("Tracking lost: not enough valid points.")
            self.reset()
            return None
        self.prev_gray = frame_gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)
        tracked_mask = self.build_mask_from_points(good_new, frame_gray.shape)
        self.last_mask = tracked_mask
        if hsv_mask is not None:
            tracked_mask = self.refine_with_hsv(tracked_mask, hsv_mask)
        return tracked_mask

    def build_mask_from_points(self, points, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if points is None or len(points) < 3:
            return mask
        hull = cv2.convexHull(points.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        return mask

    def refine_with_hsv(self, tracked_mask, hsv_mask):
        if hsv_mask is None:
            return tracked_mask
        overlap = cv2.bitwise_and(tracked_mask, hsv_mask)
        if np.count_nonzero(overlap) > 0:
            return overlap
        return cv2.bitwise_or(tracked_mask, hsv_mask)

    @property
    def point_count(self):
        return 0 if self.prev_points is None else len(self.prev_points)


def load_settings(settings_path):
    if not os.path.exists(settings_path):
        return {}
    try:
        with open(settings_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            if isinstance(data, dict):
                return data
            print(f"Settings file {settings_path} did not contain a JSON object. Using defaults.")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to load settings from {settings_path}: {exc}")
    return {}


def save_settings(settings_path, values):
    try:
        with open(settings_path, "w", encoding="utf-8") as outfile:
            json.dump(values, outfile, indent=2)
        print(f"Saved settings to {settings_path}")
    except OSError as exc:
        print(f"Failed to save settings to {settings_path}: {exc}")


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


def build_overlay_lines(looper, lower_hue, upper_hue, show_original, apply_morph, apply_smooth, apply_components, recording, tracking_enabled, tracker_active, tracker_points, hsv_refine, fill_green):
    current = looper.current() if looper else "None"
    total = len(looper.available()) if looper else 0
    position = ((looper.index % total) + 1) if looper and total else 0
    count_label = f"{position}/{total}" if total else "0/0"
    tracking_label = "ACTIVE" if tracker_active else ("ARMED" if tracking_enabled else "OFF")
    lines = [
        f"Hue: {lower_hue}-{upper_hue} (h/j/k/l)",
        f"Overlay: {'ON' if show_original else 'OFF'} (o)",
        f"Morph: {'ON' if apply_morph else 'OFF'} (m)",
        f"Smooth: {'ON' if apply_smooth else 'OFF'} (g)",
        f"Components: {'ON' if apply_components else 'OFF'} (c)",
    f"Tracking: {tracking_label} (t)",
    f"HSV refine: {'ON' if hsv_refine else 'OFF'} (f)",
    f"Tracked pts: {tracker_points}",
    f"Fill green: {'ON' if fill_green else 'OFF'} (b)",
        f"Recording: {'ON' if recording else 'OFF'} (v)",
        f"Video: {current if current else 'None'} ({count_label})",
        "p: save settings",
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


lower_hue = 30
upper_hue = 90
lower_sat = 50
upper_sat = 255
lower_val = 60
upper_val = 255
hue_step = 2

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
VIDEO_OUT_DIR = os.path.join(BASE_DIR, "video_output")
SETTINGS_PATH = os.path.join(BASE_DIR, "projection_settings.json")
FILL_COLOR = (0, 255, 0)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    sys.exit(1)

video_looper = None
show_original_video = True
show_instructions = True
apply_morphology = False
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
apply_smoothing = False
smooth_kernel = (5, 5)
apply_component_filter = False
min_component_area = 800
recording_output = None
recording_writer = None
use_tracking = False
use_hsv_refine = True
fill_green = False
tracker = OpticalFlowTracker()

loaded_settings = load_settings(SETTINGS_PATH)
if loaded_settings:
    lower_hue = int(loaded_settings.get("lower_hue", lower_hue))
    upper_hue = int(loaded_settings.get("upper_hue", upper_hue))
    lower_sat = int(loaded_settings.get("lower_sat", lower_sat))
    upper_sat = int(loaded_settings.get("upper_sat", upper_sat))
    lower_val = int(loaded_settings.get("lower_val", lower_val))
    upper_val = int(loaded_settings.get("upper_val", upper_val))
    show_original_video = bool(loaded_settings.get("show_original_video", show_original_video))
    show_instructions = bool(loaded_settings.get("show_instructions", show_instructions))
    apply_morphology = bool(loaded_settings.get("apply_morphology", apply_morphology))
    apply_smoothing = bool(loaded_settings.get("apply_smoothing", apply_smoothing))
    apply_component_filter = bool(loaded_settings.get("apply_component_filter", apply_component_filter))
    min_component_area = int(loaded_settings.get("min_component_area", min_component_area))
    hue_step = int(loaded_settings.get("hue_step", hue_step))
    use_hsv_refine = bool(loaded_settings.get("use_hsv_refine", use_hsv_refine))
    fill_green = bool(loaded_settings.get("fill_green", fill_green))
    print(f"Loaded settings from {SETTINGS_PATH}")

try:
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        if video_looper is None:
            video_looper = VideoLooper(video_dir=os.path.join(os.path.dirname(__file__), 'videos'), frame_size=(w, h))
        else:
            if video_looper.w != w or video_looper.h != h:
                video_looper.set_frame_size((w, h))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if apply_smoothing:
            hsv = cv2.GaussianBlur(hsv, smooth_kernel, 0)
        lower_bounds = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
        upper_bounds = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

        if apply_morphology:
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        if apply_component_filter:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hsv_mask, connectivity=8)
            filtered = np.zeros_like(hsv_mask)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                    filtered[labels == label] = 255
            hsv_mask = filtered

        tracked_mask = None
        if use_tracking:
            if tracker.active:
                tracked_mask = tracker.update(gray, hsv_mask if use_hsv_refine else None)
                if tracked_mask is None:
                    print("Tracking disabled: lost target.")
                    use_tracking = False
            else:
                success = tracker.initialize(gray, hsv_mask)
                if not success:
                    use_tracking = False

        final_mask = tracked_mask if tracked_mask is not None else hsv_mask

        vid_frame = video_looper.next_frame() if video_looper else None
        if vid_frame is None:
            vid_frame = np.zeros_like(frame)

        if show_original_video:
            result = frame.copy()
        else:
            result = np.zeros_like(frame)
        if fill_green:
            result[final_mask > 0] = FILL_COLOR
        else:
            result[final_mask > 0] = vid_frame[final_mask > 0]

        overlay_lines = build_overlay_lines(
            video_looper,
            lower_hue,
            upper_hue,
            show_original_video,
            apply_morphology,
            apply_smoothing,
            apply_component_filter,
            recording_writer is not None,
            use_tracking,
            tracker.active,
            tracker.point_count,
            use_hsv_refine,
            fill_green,
        ) if show_instructions else []
        result = draw_overlay(result, overlay_lines)

        if recording_writer is not None:
            recording_writer.write(result)

        cv2.imshow('Result', result)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
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
        elif key == ord('t'):
            if use_tracking:
                print("Tracking deactivated by user.")
                use_tracking = False
                tracker.reset()
            else:
                if tracker.initialize(gray, hsv_mask):
                    use_tracking = True
                else:
                    print("Press t again after adjusting the mask to initialize tracking.")
        elif key == ord('f'):
            use_hsv_refine = not use_hsv_refine
            print(f"HSV refinement {'enabled' if use_hsv_refine else 'disabled'}.")
        elif key == ord('b'):
            fill_green = not fill_green
            print(f"Green fill {'enabled' if fill_green else 'disabled'}.")
        elif key == ord('p'):
            current_settings = {
                "lower_hue": int(lower_hue),
                "upper_hue": int(upper_hue),
                "lower_sat": int(lower_sat),
                "upper_sat": int(upper_sat),
                "lower_val": int(lower_val),
                "upper_val": int(upper_val),
                "show_original_video": bool(show_original_video),
                "show_instructions": bool(show_instructions),
                "apply_morphology": bool(apply_morphology),
                "apply_smoothing": bool(apply_smoothing),
                "apply_component_filter": bool(apply_component_filter),
                "min_component_area": int(min_component_area),
                "hue_step": int(hue_step),
                "use_hsv_refine": bool(use_hsv_refine),
                "fill_green": bool(fill_green),
            }
            save_settings(SETTINGS_PATH, current_settings)
        elif key == ord('n'):
            if video_looper:
                video_looper.next_video()
        elif key == ord('o'):
            show_original_video = not show_original_video
        elif key == ord('m'):
            apply_morphology = not apply_morphology
        elif key == ord('g'):
            apply_smoothing = not apply_smoothing
        elif key == ord('c'):
            apply_component_filter = not apply_component_filter
        elif key == ord('v'):
            if recording_writer is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(VIDEO_OUT_DIR, f"projection_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                recording_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (result.shape[1], result.shape[0]))
                if not recording_writer.isOpened():
                    print(f"Failed to start recording at {video_path}")
                    recording_writer = None
                    recording_output = None
                else:
                    recording_output = video_path
                    print(f"Recording started: {video_path}")
            else:
                print(f"Recording stopped: {recording_output}")
                recording_writer.release()
                recording_writer = None
                recording_output = None
        elif key == ord('r'):
            if video_looper:
                video_looper.reload()
        elif key == ord('s'):
            snapshot_name = f"projection_snapshot_{int(time.time())}.png"
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
            cv2.imwrite(snapshot_path, result)
            print(f"Saved snapshot {snapshot_path}")
        elif key == ord('?'):
            show_instructions = not show_instructions

finally:
    cap.release()
    cv2.destroyAllWindows()
    if recording_writer is not None:
        recording_writer.release()
    if video_looper:
        video_looper.cleanup()
    tracker.reset()
    print("Cleaned up. Bye.")
