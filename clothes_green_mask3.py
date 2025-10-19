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
        # additional toggles
        self.show_noise = True
        self.show_background = True
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
                print("  t                                - toggle noise fill on/off")
                print("  v                                - toggle background video on/off")
                print("  quit                             - exit program")
                print("  help                             - this message")
            elif cmd == "quit":
                print("Quitting per user request.")
                self.running = False
            elif cmd == "t":
                with self.lock:
                    self.show_noise = not self.show_noise
                    state = "ON" if self.show_noise else "OFF"
                print(f"Noise fill toggled: {state}")
            elif cmd == "v":
                with self.lock:
                    self.show_background = not self.show_background
                    state = "ON" if self.show_background else "OFF"
                print(f"Background video toggled: {state}")
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

    def get_show_noise(self):
        with self.lock:
            return self.show_noise

    def get_show_background(self):
        with self.lock:
            return self.show_background


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

# Pre-read first frame to set up animated noise field grids
ret_first, _frame_first = cap.read()
if not ret_first:
    print("Error: cannot read initial frame for noise setup")
    cap.release()
    sys.exit(1)
height, width = _frame_first.shape[:2]
########################################
# Perlin-like procedural noise generator
########################################

class PerlinField:
    """Fast-ish 2D periodic Perlin-style noise sampled on a low-res grid and upscaled.
    Generates values in [0,1]. Uses 2 fractal octaves.
    Option A: analytic range mapping (avoid min/max collapse).
    Option B: increased variation (fewer base cells + larger scales).
    """
    def __init__(self, base_cells=32, low_res=(160, 90), seed=None):
        self.base_cells = base_cells  # number of gradient cells per axis (periodic)
        self.low_w, self.low_h = low_res
        rng = np.random.default_rng(seed)
        # gradient vectors (normalized) periodic
        angles = rng.uniform(0, 2*np.pi, size=(base_cells, base_cells))
        self.gradients = np.stack([np.cos(angles), np.sin(angles)], axis=2).astype(np.float32)
        # coordinate template low-res (will animate via offsets)
        self.x_coords = np.linspace(0, base_cells, self.low_w, dtype=np.float32)[None, :]
        self.y_coords = np.linspace(0, base_cells, self.low_h, dtype=np.float32)[:, None]

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)  # 6t^5 -15t^4 +10t^3

    def _single_layer(self, off_x, off_y, scale):
        # Apply offsets and scale (separate 1D axes)
        X1d = (self.x_coords[0] + off_x) * scale            # shape (low_w,)
        Y1d = (self.y_coords[:,0] + off_y) * scale           # shape (low_h,)

        # Integer grid coordinates (periodic)
        gx_base = (np.floor(X1d).astype(int)) % self.base_cells  # (low_w,)
        gy_base = (np.floor(Y1d).astype(int)) % self.base_cells  # (low_h,)
        gx1_base = (gx_base + 1) % self.base_cells
        gy1_base = (gy_base + 1) % self.base_cells

        # Fractional positions
        xf1d = X1d - np.floor(X1d)  # (low_w,)
        yf1d = Y1d - np.floor(Y1d)  # (low_h,)

        # Broadcast to 2D grids
        xf = np.broadcast_to(xf1d, (self.low_h, self.low_w))  # (H,W)
        yf = np.broadcast_to(yf1d[:,None], (self.low_h, self.low_w))
        u = self._fade(xf)
        v = self._fade(yf)

        # 2D index grids for gradients
        GX, GY = np.broadcast_to(gx_base, (self.low_h, self.low_w)), np.broadcast_to(gy_base[:,None], (self.low_h, self.low_w))
        GX1, GY1 = np.broadcast_to(gx1_base, (self.low_h, self.low_w)), np.broadcast_to(gy1_base[:,None], (self.low_h, self.low_w))

        # Fetch gradient vectors per corner
        g00 = self.gradients[GY,  GX ]            # (H,W,2)
        g10 = self.gradients[GY,  GX1]
        g01 = self.gradients[GY1, GX ]
        g11 = self.gradients[GY1, GX1]

        # Offset vectors for dot products
        d00 = np.stack([xf, yf], axis=2)
        d10 = np.stack([xf-1, yf], axis=2)
        d01 = np.stack([xf, yf-1], axis=2)
        d11 = np.stack([xf-1, yf-1], axis=2)

        dot00 = np.sum(g00 * d00, axis=2)
        dot10 = np.sum(g10 * d10, axis=2)
        dot01 = np.sum(g01 * d01, axis=2)
        dot11 = np.sum(g11 * d11, axis=2)

        lerp_x0 = dot00 + u * (dot10 - dot00)
        lerp_x1 = dot01 + u * (dot11 - dot01)
        value = lerp_x0 + v * (lerp_x1 - lerp_x0)
        return value  # roughly in [-1,1]

    def generate(self, t):
        # Two octaves with larger scales for more visible variation (Option B)
        layer1 = self._single_layer(off_x=t*0.35, off_y=t*0.27, scale=1.0)
        layer2 = self._single_layer(off_x=t*0.78+50, off_y=t*0.63+25, scale=2.0)
        combo = layer1*0.6 + layer2*0.4  # still roughly in [-1,1]
        # Analytic mapping from [-1,1] -> [0,1] (Option A)
        combo = combo * 0.5 + 0.5
        # Mild contrast boost (keep values in [0,1])
        combo = np.clip((combo - 0.5) * 1.3 + 0.5, 0.0, 1.0)
        return combo

perlin = PerlinField(base_cells=32, low_res=(min(240, width//2), min(135, height//2)))

print("Controls: ESC quit | terminal: 't' noise, 'v' background, 'show' thresholds, 'quit' exit")

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

        # Decide base frame depending on background toggle
        if controls.get_show_background():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        if controls.get_show_noise() and np.any(mask):
            t = time.time()
            base = perlin.generate(t)
            # Upscale to frame size
            up = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
            # Create subtle colorization by phase-shifting
            r = up
            g = np.roll(up, shift=5, axis=0)  # vertical phase shift
            b = np.roll(up, shift=11, axis=1) # horizontal phase shift
            noise_img = np.stack([
                (b*255).astype(np.uint8),
                (g*255).astype(np.uint8),
                (r*255).astype(np.uint8)
            ], axis=2)
            noise_img = cv2.GaussianBlur(noise_img, (0, 0), 0.8)
            result[mask > 0] = noise_img[mask > 0]
        else:
            # Fallback flat color if noise disabled or mask empty
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