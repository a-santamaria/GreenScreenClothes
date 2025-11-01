import cv2
import numpy as np
import sys
import time
import math
import random
import threading
import queue

try:
    import noise as noise_lib
except ImportError:
    raise SystemExit("Please install the 'noise' package (pip install noise) or add it to requirements.txt")


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


class PerlinFieldGenerator:
    """Produces a procedural Perlin-noise-based color field with runtime controls."""

    def __init__(self, width, height, cell_size=3):
        self.width = int(width)
        self.height = int(height)
        self.cell_size = max(1, int(cell_size))
        self.cols = int(math.ceil(self.width / self.cell_size))
        self.rows = int(math.ceil(self.height / self.cell_size))

        # Noise parameters
        self.seed = 42
        self.octaves = 5
        self.persistence = 0.5
        self.lacunarity = 2.0
        self.base_scale = 0.8
        self.time_scale = 0.2
        self.time_offset = 0.0

        # Rendering toggles
        self.use_points = False
        self.use_segments = True
        self.use_interpolation = False
        self.use_feather = False
        self.apply_blur = False

        self.palette_cycle = ["grayscale", "ember", "earth"]
        self.palette_mode = 1
        self.sample_stride = 2

        # Palette configuration used by the runtime controls
        self.palette_definitions = {
            "ember": [
                (27, 1, 103),    # #67011b
                (9, 42, 199),    # #c72a09
                (103, 46, 1),    # #012e67
                (5, 166, 225),   # #e1a605
                (69, 41, 89),    # #592945
                (39, 38, 37),    # #252627
            ],
            "earth": [
                (35, 54, 82),
                (57, 96, 142),
                (68, 120, 160),
                (90, 153, 180),
                (110, 189, 215),
                (110, 189, 215),
            ],
        }
        self.segment_specs = {
            "ember": [25, 20, 15, 15, 15, 10],
            "earth": [20, 20, 20, 20, 10, 10],
        }

        # Cached canvas so we can reuse noise between frames
        self.cached_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.noise_refresh_interval = 10.0
        self._time_since_refresh = self.noise_refresh_interval
        self._pending_time = 0.0
        self.last_region = None

    def _mark_dirty(self):
        """Force a recomputation on next generate_frame call."""
        self._time_since_refresh = self.noise_refresh_interval

    def _color_from_brightness(self, value):
        """Return a BGR tuple corresponding to a normalized brightness value."""
        normalized = max(0.0, min(0.999999, value))

        if self.palette_mode == 0:
            val = int(round(normalized * 255))
            return (val, val, val)

        palette_key = self.palette_cycle[self.palette_mode]
        palette = self.palette_definitions[palette_key]
        palette_len = len(palette)

        bin_idx = 0
        prev_idx = None
        next_idx = None
        position = 0.0

        if self.use_segments:
            spec = self.segment_specs.get(palette_key)
            if not spec:
                spec = [1] * palette_len
            total_segments = float(sum(spec))
            segment_value = normalized * total_segments
            if segment_value >= total_segments:
                segment_value = total_segments - 1e-6

            cumulative = 0.0
            for i, seg_count in enumerate(spec):
                next_cumulative = cumulative + seg_count
                if segment_value < next_cumulative or i == len(spec) - 1:
                    bin_idx = min(i, palette_len - 1)
                    band_start = cumulative / total_segments
                    band_end = next_cumulative / total_segments
                    position = (normalized - band_start) / max(1e-6, band_end - band_start)
                    prev_idx = i - 1 if i > 0 else None
                    next_idx = i + 1 if i < palette_len - 1 else None
                    break
                cumulative = next_cumulative
        else:
            loops = 3
            total_bins = palette_len * loops
            bin_value = normalized * total_bins
            if bin_value >= total_bins:
                bin_value = total_bins - 1e-6
            bin_loop_idx = int(math.floor(bin_value))
            position = bin_value - bin_loop_idx
            bin_idx = bin_loop_idx % palette_len
            if palette_len > 1:
                prev_idx = (bin_idx - 1) % palette_len
                next_idx = (bin_idx + 1) % palette_len

        color_vec = np.array(palette[bin_idx], dtype=np.float32)

        if self.use_interpolation and next_idx is not None:
            next_color = np.array(palette[next_idx], dtype=np.float32)
            color_vec = color_vec * (1.0 - position) + next_color * position

        if self.use_feather:
            feather_ratio = 0.2
            if position < feather_ratio and prev_idx is not None:
                t = position / feather_ratio
                prev_color = np.array(palette[prev_idx], dtype=np.float32)
                color_vec = prev_color * (1.0 - t) + color_vec * t
            elif position > 1.0 - feather_ratio and next_idx is not None:
                t = (position - (1.0 - feather_ratio)) / feather_ratio
                t = max(0.0, min(1.0, t))
                next_color = np.array(palette[next_idx], dtype=np.float32)
                color_vec = color_vec * (1.0 - t) + next_color * t

        return tuple(int(np.clip(round(v), 0, 255)) for v in color_vec)

    def generate_frame(self, dt, region=None):
        """Return a noise frame, recomputing only as needed in the requested region."""
        if dt is None:
            dt = 0.0

        region_tuple = None
        if region is not None:
            x0, y0, x1, y1 = region
            region_tuple = (
                max(0, int(x0)),
                max(0, int(y0)),
                min(self.width, int(x1)),
                min(self.height, int(y1)),
            )
            if region_tuple[2] <= region_tuple[0] or region_tuple[3] <= region_tuple[1]:
                region_tuple = None

        self._pending_time += dt
        self._time_since_refresh += dt

        if region_tuple is None:
            if self.cached_canvas is None:
                self.cached_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                self.cached_canvas.fill(0)
            self.last_region = None
            self._pending_time = 0.0
            self._time_since_refresh = 0.0
            if self.apply_blur:
                return cv2.GaussianBlur(self.cached_canvas, (5, 5), 0)
            return self.cached_canvas

        needs_update = False
        if self.cached_canvas is None:
            self.cached_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            needs_update = True
        if region_tuple != self.last_region:
            needs_update = True
        if self._time_since_refresh >= self.noise_refresh_interval:
            needs_update = True

        if needs_update:
            self.time_offset += self._pending_time * self.time_scale
            self._pending_time = 0.0
            self._time_since_refresh = 0.0
            self._compute_canvas(region_tuple)
            self.last_region = region_tuple

        if self.apply_blur:
            return cv2.GaussianBlur(self.cached_canvas, (5, 5), 0)
        return self.cached_canvas

    def _compute_canvas(self, region):
        """Recompute the Perlin field inside region (or the full frame when None)."""
        if self.cached_canvas is None:
            self.cached_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if region is None:
            x_start, y_start, x_end, y_end = 0, 0, self.width, self.height
        else:
            margin = int(self.cell_size * self.sample_stride)
            x_start = max(0, region[0] - margin)
            y_start = max(0, region[1] - margin)
            x_end = min(self.width, region[2] + margin)
            y_end = min(self.height, region[3] + margin)

        step_px = max(1, self.cell_size * self.sample_stride)

        for y0 in range(y_start, y_end, step_px):
            y1 = min(y_end, y0 + step_px)
            if y1 <= y0:
                continue
            y_center = y0 + (y1 - y0) * 0.5

            for x0 in range(x_start, x_end, step_px):
                x1 = min(x_end, x0 + step_px)
                if x1 <= x0:
                    continue
                x_center = x0 + (x1 - x0) * 0.5

                nx = x_center * self.base_scale / 100.0
                ny = y_center * self.base_scale / 100.0

                n = noise_lib.pnoise3(
                    nx,
                    ny,
                    self.time_offset,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    repeatz=1024,
                    base=self.seed,
                )

                normalized = (n + 1.0) * 0.5
                color = self._color_from_brightness(normalized)

                if self.use_points:
                    sub_view = self.cached_canvas[y0:y1, x0:x1]
                    sub_view[:] = (0, 0, 0)
                    center = (max(0, sub_view.shape[1] // 2), max(0, sub_view.shape[0] // 2))
                    radius = max(1, min(sub_view.shape[0], sub_view.shape[1]) // 2)
                    cv2.circle(sub_view, center, radius, color, -1, cv2.LINE_AA)
                else:
                    self.cached_canvas[y0:y1, x0:x1] = color

    def handle_key(self, key):
        """Handle keyboard shortcuts specific to the Perlin field."""
        handled = True

        if key == ord('c'):
            self.palette_mode = (self.palette_mode + 1) % len(self.palette_cycle)
            print(f"Color mode: {self.palette_cycle[self.palette_mode]}")
            self._mark_dirty()
        elif key == ord('x'):
            self.use_points = not self.use_points
            print(f"Draw mode: {'Points' if self.use_points else 'Rectangles'}")
            self._mark_dirty()
        elif key == ord('u'):
            self.use_segments = not self.use_segments
            print(f"Segments: {'ON' if self.use_segments else 'OFF'}")
            self._mark_dirty()
        elif key == ord('i'):
            self.use_interpolation = not self.use_interpolation
            print(f"Interpolation: {'ON' if self.use_interpolation else 'OFF'}")
            self._mark_dirty()
        elif key == ord('f'):
            self.use_feather = not self.use_feather
            print(f"Feather: {'ON' if self.use_feather else 'OFF'}")
            self._mark_dirty()
        elif key == ord('b'):
            self.apply_blur = not self.apply_blur
            print(f"Post blur: {'ON' if self.apply_blur else 'OFF'}")
        elif key == ord('1'):
            self.sample_stride = min(8, self.sample_stride + 1)
            print(f"Sampling stride (coarser): {self.sample_stride}")
            self._mark_dirty()
        elif key == ord('2'):
            self.sample_stride = max(1, self.sample_stride - 1)
            print(f"Sampling stride (finer): {self.sample_stride}")
            self._mark_dirty()
        elif key in (ord('+'), ord('=')):
            self.base_scale *= 1.15
            print(f"Scale increased: {self.base_scale:.3f}")
            self._mark_dirty()
        elif key in (ord('-'), ord('_')):
            self.base_scale = max(0.05, self.base_scale / 1.15)
            print(f"Scale decreased: {self.base_scale:.3f}")
            self._mark_dirty()
        elif key == ord(']'):
            self.time_scale = min(5.0, self.time_scale * 1.2)
            print(f"Time scale increased: {self.time_scale:.3f}")
            self._mark_dirty()
        elif key == ord('['):
            self.time_scale = max(0.01, self.time_scale / 1.2)
            print(f"Time scale decreased: {self.time_scale:.3f}")
            self._mark_dirty()
        elif key == ord('r'):
            self.seed = random.randint(0, 10_000_000)
            print(f"Noise reseeded: {self.seed}")
            self._mark_dirty()
        else:
            handled = False

        return handled

    def handle_command(self, command):
        """Translate terminal commands into the existing key handling."""
        if not command:
            return False

        normalized = command.strip().lower()
        if not normalized:
            return False

        alias_map = {
            "palette": 'c',
            "color": 'c',
            "points": 'x',
            "segments": 'u',
            "interp": 'i',
            "interpolation": 'i',
            "feather": 'f',
            "blur": 'b',
            "stride+": '1',
            "stride-": '2',
            "scale+": '+',
            "scale-": '-',
            "time+": ']',
            "time-": '[',
            "reseed": 'r',
        }

        if normalized in alias_map:
            return self.handle_key(ord(alias_map[normalized]))

        if len(normalized) == 1:
            return self.handle_key(ord(normalized))

        return False

    def get_overlay_lines(self):
        """Return overlay text describing current Perlin controls."""
        palette_label = self.palette_cycle[self.palette_mode]
        lines = [
            f"Palette: {palette_label} (c)",
            f"Points: {'ON' if self.use_points else 'OFF'} (x) | Segments: {'ON' if self.use_segments else 'OFF'} (u)",
            f"Interp: {'ON' if self.use_interpolation else 'OFF'} (i) | Feather: {'ON' if self.use_feather else 'OFF'} (f)",
            f"Blur: {'ON' if self.apply_blur else 'OFF'} (b) | Stride: {self.sample_stride} (1/2)",
            f"Scale: {self.base_scale:.2f} (+/-) | Time: {self.time_scale:.2f} ([/])",
            "r: reseed noise",
        ]
        return lines


def print_terminal_instructions():
    print()
    print("Type a command and press Enter. Available commands:")
    instructions = [
        "h : decrease lower hue",
        "j : increase lower hue",
        "k : decrease upper hue",
        "l : increase upper hue",
        "c : cycle color palette",
        "x : toggle points",
        "u : toggle segments",
        "i : toggle interpolation",
        "f : toggle feather",
        "b : toggle blur",
        "1 : coarser sampling stride",
        "2 : finer sampling stride",
        "+ / - : adjust scale",
        "[ / ] : adjust time scale",
        "r : reseed noise",
        "video : toggle original feed",
        "status : print current settings",
        "help : show this message",
        "quit : exit",
    ]
    for line in instructions:
        print(f"  {line}")
    print()


def print_status(perlin_field, lower_hue, upper_hue, show_original_video):
    print(f"Hue range: {lower_hue}-{upper_hue}")
    print(f"Original video: {'ON' if show_original_video else 'OFF'}")
    if perlin_field is None:
        print("Perlin field not initialized yet.")
        return
    for line in perlin_field.get_overlay_lines():
        print(line)


def handle_hsv_command(command, lower_hue, upper_hue, step):
    """Adjust HSV thresholds based on terminal commands."""
    updated = False

    if command == 'h':
        if lower_hue > 0:
            lower_hue = max(0, lower_hue - step)
            if lower_hue >= upper_hue:
                upper_hue = min(179, lower_hue + 1)
            updated = True
    elif command == 'j':
        if lower_hue < upper_hue - 1:
            lower_hue = min(upper_hue - 1, lower_hue + step)
            updated = True
    elif command == 'k':
        if lower_hue < upper_hue - 1:
            upper_hue = max(lower_hue + 1, upper_hue - step)
            updated = True
    elif command == 'l':
        if upper_hue < 179:
            upper_hue = min(179, upper_hue + step)
            updated = True

    return lower_hue, upper_hue, updated


# Initial HSV thresholds
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

perlin_field = None
prev_time = time.time()
show_original_video = True

try:
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_height, frame_width = frame.shape[:2]

        if perlin_field is None or perlin_field.width != frame_width or perlin_field.height != frame_height:
            perlin_field = PerlinFieldGenerator(frame_width, frame_height, cell_size=3)

        # Process any pending terminal commands before rendering the frame.
        while True:
            command = command_listener.get_nowait()
            if command is None:
                break
            lowered = command.lower()
            if lowered in {"quit", "exit", "q"}:
                print("Quitting (terminal command).")
                running = False
                break
            if lowered in {"help", "?"}:
                print_terminal_instructions()
                continue
            if lowered == "status":
                print_status(perlin_field, lower_hue, upper_hue, show_original_video)
                continue
            if lowered in {"video", "original"}:
                show_original_video = not show_original_video
                state = "ON" if show_original_video else "OFF"
                print(f"Original video toggled: {state}")
                continue

            lower_hue, upper_hue, hsv_updated = handle_hsv_command(lowered, lower_hue, upper_hue, hue_step)
            if hsv_updated:
                print(f"Hue range updated: {lower_hue}-{upper_hue}")
                continue

            handled = False
            if perlin_field is not None:
                handled = perlin_field.handle_command(command)

            if not handled:
                print(f"Unknown command: {command}")

        if not running:
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bounds = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
        upper_bounds = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

        region = None
        nonzero = cv2.findNonZero(mask)
        if nonzero is not None:
            x, y, w, h = cv2.boundingRect(nonzero)
            region = (x, y, x + w, y + h)

        perlin_frame = perlin_field.generate_frame(dt, region=region)

        if show_original_video:
            result = frame.copy()
        else:
            result = np.zeros_like(frame)
        result[mask > 0] = perlin_frame[mask > 0]

        cv2.imshow("Result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Quitting (ESC pressed).")
            break
        elif key == ord('q'):
            print("Quitting (Q pressed).")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    command_listener.stop()
    print("Cleaned up. Bye.")