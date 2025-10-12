import cv2
import numpy as np
import threading
import time
import sys

def clamp(v, lo, hi):
    try:
        v = int(v)
    except:
        return None
    return max(lo, min(hi, v))

class TerminalControls:
    """
    Runs a background thread reading stdin commands to update HSV thresholds:
    Commands:
      - set LH 35
      - set US 200
      - show
      - help
      - quit
    """
    def __init__(self):
        self.lock = threading.Lock()
        # default values
        # Hue 0–179 : color 
        # Saturation 0–255 : color intensity
        # Value 0–255 : brightness
        self.values = {
            "LH": 30, "LS":50, "LV": 60,
            "UH": 90, "US": 255, "UV": 255
        }
        self.running = True
        self.thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self.thread.start()

    def _stdin_loop(self):
        print("Terminal controls started. Type 'help' for commands.")
        while self.running:
            try:
                line = input()  # blocking, but in background thread
            except EOFError:
                # Input closed, stop
                self.running = False
                break
            except Exception as e:
                print("Input error:", e)
                continue

            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "set" and len(parts) == 3:
                key = parts[1].upper()
                if key not in self.values:
                    print(f"Unknown key: {key}. Valid keys: {', '.join(self.values.keys())}")
                    continue
                val = clamp(parts[2], 0, 255 if key != "LH" and key != "UH" else 179)
                if val is None:
                    print("Value must be an integer.")
                    continue
                with self.lock:
                    self.values[key] = val
                print(f"Set {key} = {val}")
            elif cmd == "show":
                with self.lock:
                    print("Current thresholds:", self.values)
            elif cmd == "help":
                print("Commands:")
                print("  set <LH|LS|LV|UH|US|UV> <value>  - set a threshold (LH/UH in 0-179, others 0-255)")
                print("  show                             - show current thresholds")
                print("  quit                             - exit program")
                print("  help                             - this message")
            elif cmd == "quit":
                print("Quitting per user request.")
                self.running = False
            else:
                print("Unknown command. Type 'help' for list of commands.")

    def get_thresholds(self):
        with self.lock:
            # return copies (ints)
            return (
                self.values["LH"],
                self.values["LS"],
                self.values["LV"],
                self.values["UH"],
                self.values["US"],
                self.values["UV"]
            )

    def stop(self):
        self.running = False
        # In case input() is blocked and we can't easily interrupt it, we exit anyway when program terminates.
        self.thread.join(timeout=1.0)


def nothing(x): pass

# Initialize terminal controls
controls = TerminalControls()

# initialize capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    sys.exit(1)

# Optionally reduce resolution to make processing faster (tweak as needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Result", 800, 600)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read thresholds from terminal-controlled variables
        lh, ls, lv, uh, us, uv = controls.get_thresholds()

        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        result = frame.copy()
        result[mask > 0] = (255, 0, 0)

        # Show result
        cv2.imshow("Result", result)

        # Keep GUI responsive. Check for ESC key to exit.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Quitting (ESC pressed).")
            break

        # Check if terminal thread requested quit
        if not controls.running:
            print("Terminal controls requested quit.")
            break

finally:
    controls.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up. Bye.")