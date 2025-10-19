import cv2
import numpy as np
import time
import random
import math

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
    octaves = 8          # single octave (can raise if desired)
    persistence = 0.5    # kept for potential future multi-octave use
    lacunarity = 2.0
    base_scale = 0.5          # spatial frequency
    time_scale = 0.3         # how fast field evolves (0 => static field)
    brightness_max = 100        # requested display max (logical scale 0..100)

    show_values = False         # overlay numeric brightness values
    use_palette = True          # toggle between earth palette and grayscale

    cv2.namedWindow("Perlin Rect Field", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Perlin Rect Field", width, height)

    print("Perlin Rectangle Field")
    print("ESC/q: quit | v: toggle numeric values | c: toggle color palette | +/-: change scale | h/l: adjust time scale | p: pause | s: screenshot | r: reseed")

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

                if use_palette:
                    # Earth-tone palette (BGR) for OpenCV: deep brown -> sand -> olive -> sage -> cream
                    palette = [
                        (35, 54, 82),    # dark soil (deep brown)
                        (57, 96, 142),   # warm brown
                        (68, 120, 160),  # olive/earth green
                        (90, 153, 180),  # sage
                        (110, 189, 215)  # light sand/sky
                    ]
                    # 15 segments across brightness range, cycling colors 1..5 three times
                    segments = 15
                    segment_size = max(1, brightness_max // segments)
                    segment_idx = min(segments - 1, brightness // segment_size)
                    bin_idx = segment_idx % 5  # cycle 0..4
                    color = palette[bin_idx]
                else:
                    # Convert to 0..255 for grayscale display (scaled by brightness_max/100)
                    display_val = int(brightness * 255 / brightness_max)
                    color = (display_val, display_val, display_val)

                x0 = c * cell_size
                y0 = r * cell_size
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_size - 1, y0 + cell_size - 1), color, -1)

                if show_values and cell_size >= 14:  # only draw text if enough room
                    txt = str(brightness)
                    text_color = (0, 0, 0)
                    if use_palette:
                        text_color = (0, 0, 0) if bin_idx < 3 else (70, 70, 70)
                    else:
                        text_color = (0, 0, 255) if brightness > brightness_max * 0.7 else (0, 0, 0)
                    cv2.putText(canvas, txt, (x0 + 2, y0 + cell_size - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, text_color, 1, cv2.LINE_AA)

        cv2.imshow("Perlin Rect Field", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC / q
            break
        elif key == ord('v'):
            show_values = not show_values
            print(f"Values overlay: {'ON' if show_values else 'OFF'}")
        elif key == ord('c'):
            use_palette = not use_palette
            mode = "Earth palette" if use_palette else "Grayscale"
            print(f"Color mode: {mode}")
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
            cv2.imwrite(fname, canvas)
            print(f"Saved {fname}")
        elif key == ord('r'):  # regenerate / change seed
            seed = random.randint(0, 10_000_000)
            print(f"Noise reseeded base={seed}")

        frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
