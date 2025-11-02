import cv2
import numpy as np
import sys
import time
import os
import threading
import queue
import glob


class TerminalCommandListener:
    """Background reader that captures commands from stdin without blocking video."""

    def __init__(self):
        self._commands = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
                continue
            command = line.strip()
            if command:
                self._commands.put(command)
            if command.lower() in {"quit", "exit", "q"}:
                break

    def get_nowait(self):
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()


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


def print_terminal_instructions():
    print()
    print("Commands:")
    lines = [
        "h/j/k/l : adjust hue range (lower-/lower+/upper-/upper+)",
        "video    : toggle original camera overlay",
        "next     : switch to next video file",
        "reload   : rescan videos folder",
        "status   : print current settings",
        "help/?   : show this help",
        "quit/exit/q : exit program",
        "Keyboard shortcuts: n(next video), o(toggle overlay), q or ESC(quit)",
    ]
    for ln in lines:
        print(f"  {ln}")
    print()


def print_status(looper, lower_hue, upper_hue, show_original):
    print(f"Hue range: {lower_hue}-{upper_hue}")
    print(f"Original overlay: {'ON' if show_original else 'OFF'} (command 'video' / key 'o')")
    if looper is None:
        print("Video looper not initialized.")
        return
    current = looper.current()
    print(f"Active video: {current if current else 'None'}")
    vids = looper.available()
    if vids:
        print("Videos:")
        for v in vids:
            mark = '*' if v == current else '-'
            print(f"  {mark} {v}")


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

command_listener = TerminalCommandListener()
print_terminal_instructions()

video_looper = None
show_original_video = True

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

        # Handle terminal commands
        while True:
            cmd = command_listener.get_nowait()
            if cmd is None:
                break
            low = cmd.lower()
            if low in {"quit", "exit", "q"}:
                print("Quitting (terminal command).")
                running = False
                break
            if low in {"help", "?"}:
                print_terminal_instructions()
                continue
            if low == "status":
                print_status(video_looper, lower_hue, upper_hue, show_original_video)
                continue
            if low in {"video", "original"}:
                show_original_video = not show_original_video
                print(f"Original overlay: {'ON' if show_original_video else 'OFF'} (toggled)")
                continue
            if low == "next":
                if video_looper: video_looper.next_video()
                continue
            if low == "reload":
                if video_looper: video_looper.reload()
                continue
            lower_hue, upper_hue, hsv_upd = handle_hsv_command(low, lower_hue, upper_hue, hue_step)
            if hsv_upd:
                print(f"Hue range updated: {lower_hue}-{upper_hue}")
                continue
            print(f"Unknown command: {cmd}")

        if not running:
            break

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

        cv2.imshow('Result', result)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            print("Quitting (key press).")
            break
        elif key == ord('n'):
            if video_looper: video_looper.next_video()
        elif key == ord('o'):
            show_original_video = not show_original_video
            print(f"Original overlay: {'ON' if show_original_video else 'OFF'} (toggled)")
        elif key == ord('r'):
            if video_looper: video_looper.reload()
        elif key == ord('s'):
            # Save current composite frame snapshot
            snapshot_name = f"projection_snapshot_{int(time.time())}.png"
            cv2.imwrite(snapshot_name, result)
            print(f"Saved snapshot {snapshot_name}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    command_listener.stop()
    if video_looper: video_looper.cleanup()
    print("Cleaned up. Bye.")