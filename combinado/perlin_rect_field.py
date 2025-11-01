import cv2
import numpy as np
import time
import random
import math
import os

try:
    import noise as noise_lib  # pip package 'noise'
except ImportError:
    raise SystemExit("Please install the 'noise' package (pip install noise) or add it to requirements.txt")


def main():
    width, height = 800, 600
    cell_size = 3              # rectangle size in pixels
    cols = width // cell_size
    rows = height // cell_size

    # Noise params (using pip 'noise' library pnoise2)
    seed = 42
    octaves = 5          # single octave (can raise if desired)
    persistence = 0.5    # kept for potential future multi-octave use
    lacunarity = 2.0
    base_scale = 0.8          # spatial frequency
    time_scale = 0.2          # how fast field evolves (0 => static field)
    brightness_max = 100        # requested display max (logical scale 0..100)

    show_values = False         # overlay numeric brightness values
    palette_mode = 1            # 0=grayscale, 1=ember, 2=earth

    palette_definitions = {
        "ember": [
            (27, 1, 103),    # #67011b (crimson plum) (vinotinto)
            (9, 42, 199),    # #c72a09 (fiery orange) (rojo anaranjado)
            (103, 46, 1),    # #012e67 (deep navy) (azul)
            (5, 166, 225),   # #e1a605 (amber gold) (amarillo)
            (69, 41, 89),    # #592945 (dusky violet) (violta)
            (39, 38, 37),    # #252627 (charcoal) (negro)
        ],
        "earth": [
            (35, 54, 82),    # dark soil (deep brown)
            (57, 96, 142),   # warm brown
            (68, 120, 160),  # olive/earth green
            (90, 153, 180),  # sage
            (110, 189, 215), # light sand/sky
            # (110, 189, 215)  # light sand/sky 2
        ]
    }
    palette_cycle = ["grayscale", "ember", "earth"]
    use_points = False
    use_segments = True
    apply_blur = False
    record_video = False
    video_writer = None
    video_filename = None
    video_fps = 30.0
    video_dir = os.path.join(os.path.dirname(__file__), "videos")

    cv2.namedWindow("Perlin Rect Field", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Perlin Rect Field", width, height)

    print("Perlin Rectangle Field")
    print("ESC/q: quit | v: toggle numeric values | c: cycle color modes | x: toggle points | u: toggle segment bands | b: toggle blur | +/-: change scale | h/l: adjust time scale | p: pause | s: screenshot | r: reseed | w: toggle video record")

    # Field is static (no temporal evolution), so we don't track time.
    paused = False
    frame_count = 0
    time_offset = 0.0
    last_time = time.time()

    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        now = time.time()
        if not paused:
            time_offset += (now - last_time) * time_scale
        last_time = now

        for r in range(rows):
            for c in range(cols):
                # Sample position center of cell (optional: add small jitter)
                x = c * cell_size + cell_size * 0.5
                y = r * cell_size + cell_size * 0.5

                # Normalize to noise space
                nx = x * base_scale / 100.0  # divide by 100 so scale param feels intuitive
                ny = (y * base_scale / 100.0)  # no temporal term => static

                # Use time-evolving 3D noise; repeatz keeps the animation looping smoothly.
                n = noise_lib.pnoise3(
                    nx, ny, time_offset,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    repeatz=1024,
                    base=seed
                )
                # Map from [-1,1] -> [0,1]
                n01 = (n + 1.0) * 0.5

                # Map to 0..brightness_max
                brightness = int(n01 * brightness_max)

                bin_idx = 0
                if palette_mode == 0:
                    # Convert to 0..255 for grayscale display (scaled by brightness_max/100)
                    display_val = int(brightness * 255 / brightness_max)
                    color = (display_val, display_val, display_val)
                else:
                    palette_key = palette_cycle[palette_mode]
                    palette = palette_definitions[palette_key]
                    normalized = brightness / max(1, brightness_max)

                    if use_segments:
                        segments = 15
                        segment_idx = min(segments - 1, int(normalized * segments))
                        bin_idx = segment_idx % len(palette)
                    else:
                        bin_idx = min(len(palette) - 1, int(normalized * len(palette)))

                    color = palette[bin_idx]

                x0 = c * cell_size
                y0 = r * cell_size
                if use_points:
                    center = (int(x0 + cell_size * 0.5), int(y0 + cell_size * 0.5))
                    radius = max(1, cell_size // 2)
                    cv2.circle(canvas, center, radius, color, 3, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (x0, y0), (x0 + cell_size - 1, y0 + cell_size - 1), color, -1)

                if show_values and cell_size >= 14 and not use_points:  # text only when rectangles large enough
                    txt = str(brightness)
                    text_color = (0, 0, 0)
                    if palette_mode == 0:
                        text_color = (0, 0, 255) if brightness > brightness_max * 0.7 else (0, 0, 0)
                    else:
                        text_color = (0, 0, 0) if bin_idx < len(palette) // 2 else (70, 70, 70)
                    cv2.putText(canvas, txt, (x0 + 2, y0 + cell_size - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, text_color, 1, cv2.LINE_AA)

        display_frame = cv2.GaussianBlur(canvas, (5, 5), 0) if apply_blur else canvas

        if record_video and video_writer is not None:
            video_writer.write(display_frame)
        else:
            cv2.imshow("Perlin Rect Field", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC / q
            break
        elif key == ord('v'):
            show_values = not show_values
            print(f"Values overlay: {'ON' if show_values else 'OFF'}")
        elif key == ord('c'):
            palette_mode = (palette_mode + 1) % len(palette_cycle)
            mode_label = palette_cycle[palette_mode]
            readable = {
                "grayscale": "Grayscale",
                "earth": "Earth palette",
                "ember": "Ember palette"
            }
            print(f"Color mode: {readable.get(mode_label, mode_label.title())}")
        elif key == ord('x'):
            use_points = not use_points
            print(f"Draw mode: {'Points' if use_points else 'Rectangles'}")
        elif key == ord('u'):
            use_segments = not use_segments
            print(f"Segment bands: {'ON' if use_segments else 'OFF'}")
        elif key == ord('b'):
            apply_blur = not apply_blur
            print(f"Post blur: {'ON' if apply_blur else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            base_scale *= 1.15
            print(f"Scale increased: {base_scale:.4f}")
        elif key == ord('-') or key == ord('_'):
            base_scale /= 1.15
            print(f"Scale decreased: {base_scale:.4f}")
        elif key == ord('l'):
            time_scale *= 1.25
            if time_scale <= 0.0:
                time_scale = 0.01
            print(f"Time scale increased: {time_scale:.4f}")
        elif key == ord('h'):
            time_scale /= 1.25
            time_scale = max(0.0, time_scale)
            print(f"Time scale decreased: {time_scale:.4f}")
        # Removed octave controls (single-layer noise only)
        elif key == ord('p'):
            paused = not paused
            print(f"Paused: {paused}")
        elif key == ord('s'):
            fname = f"perlin_rect_{int(time.time())}.png"
            frame_to_save = cv2.GaussianBlur(canvas, (5, 5), 0) if apply_blur else canvas
            cv2.imwrite(fname, frame_to_save)
            print(f"Saved {fname}")
        elif key == ord('r'):  # regenerate / change seed
            seed = random.randint(0, 10_000_000)
            print(f"Noise reseeded base={seed}")
        elif key == ord('w'):
            if record_video:
                record_video = False
                if video_writer is not None:
                    video_writer.release()
                    print(f"Stopped recording {video_filename}")
                video_writer = None
                video_filename = None
            else:
                os.makedirs(video_dir, exist_ok=True)
                video_filename = os.path.join(video_dir, f"perlin_rect_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (width, height))
                if not video_writer.isOpened():
                    print("Failed to start video recording.")
                    video_writer = None
                    video_filename = None
                else:
                    record_video = True
                    print(f"Recording video to {video_filename} (window updates paused)")

        frame_count += 1

    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
