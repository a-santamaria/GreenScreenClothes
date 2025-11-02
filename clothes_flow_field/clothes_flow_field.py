import cv2
import numpy as np
import json
import math
import os
import time
import random

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "flow_field_settings.txt")


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_settings(flow_field, show_vectors, show_controls, show_video):
    payload = flow_field.to_settings()
    payload.update({
        "show_vectors": bool(show_vectors),
        "show_controls": bool(show_controls),
        "show_video": bool(show_video),
    })
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Settings saved to {SETTINGS_FILE}")
    except OSError:
        print("Failed to write settings file")


def _float_setting(settings, key, default):
    value = settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_setting(settings, key, default):
    value = settings.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value, low, high):
    return max(low, min(value, high))


MIN_PARTICLES = 0
MAX_PARTICLES = 600

FLOW_MODE_OPTIONS = [
    ("perlin", "Perlin Noise"),
    ("gradient", "Gradient Vectors"),
    ("canny", "Canny + Orientation"),
    ("structure", "Structure Tensor"),
    ("hough", "Hough Segments"),
    ("edge_filter", "Edge-Aware Filter"),
]

FLOW_MODE_KEYS = [mode for mode, _ in FLOW_MODE_OPTIONS]
FLOW_MODE_LABELS = {mode: label for mode, label in FLOW_MODE_OPTIONS}

class PerlinNoise:
    """
    Simple Perlin noise implementation for smooth, natural-looking flow fields.
    """
    
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate permutation table
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p += self.p  # Duplicate to avoid overflow
    
    def fade(self, t):
        """Smooth fade function: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a, b, t):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y):
        """Calculate gradient vector"""
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y
    
    def noise(self, x, y):
        """Generate 2D Perlin noise at coordinates (x, y)"""
        # Find grid cell coordinates
        X = int(x) & 255
        Y = int(y) & 255
        
        # Find relative coordinates within cell
        x -= int(x)
        y -= int(y)
        
        # Compute fade curves
        u = self.fade(x)
        v = self.fade(y)
        
        # Hash coordinates of 4 cube corners
        A = self.p[X] + Y
        AA = self.p[A]
        AB = self.p[A + 1]
        B = self.p[X + 1] + Y
        BA = self.p[B]
        BB = self.p[B + 1]
        
        # Blend results from 4 corners
        return self.lerp(
            self.lerp(
                self.grad(self.p[AA], x, y),
                self.grad(self.p[BA], x - 1, y),
                u
            ),
            self.lerp(
                self.grad(self.p[AB], x, y - 1),
                self.grad(self.p[BB], x - 1, y - 1),
                u
            ),
            v
        )
    
    def octave_noise(self, x, y, octaves=4, persistence=0.5, scale=1.0):
        """Generate fractal noise with multiple octaves"""
        value = 0.0
        amplitude = 1.0
        frequency = scale
        max_value = 0.0
        
        for _ in range(octaves):
            value += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return value / max_value

class FlowField:
    """
    Creates a dynamic flow field that guides particles through force vectors.
    Simulates natural phenomena like wind, water currents, or magnetic fields.
    """
    
    def __init__(self, width=800, height=600, settings=None):
        self.width = width
        self.height = height
        settings = settings or {}
        
        # Initialize Perlin noise generator
        self.noise = PerlinNoise(seed=42)
        
        # Flow field grid properties
        self.grid_size = 20  # Size of each cell in the flow field
        self.cols = width // self.grid_size + 1
        self.rows = height // self.grid_size + 1
        
        # Initialize flow field vectors
        self.flow_field = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.time_offset = 0.0
        
        # Noise parameters
        self.noise_scale = _float_setting(settings, "noise_scale", 0.01)  # Smaller values => smoother field variation
        self.time_speed = _float_setting(settings, "time_speed", 0.5)    # Controls how quickly the flow evolves over time
        self.force_magnitude = _float_setting(settings, "force_magnitude", 0.4)  # Base strength applied from the flow field
        self.curl_strength = _float_setting(settings, "curl_strength", 0.5)  # Additional curl applied for swirling motion
        self.speed_factor = _float_setting(settings, "speed_factor", 0.5)   # Global multiplier for particle speed
        self.base_trail_thickness = _float_setting(settings, "base_trail_thickness", 3.0)  # Controls visual thickness of trails
        self.trail_length_min = max(1, _int_setting(settings, "trail_length_min", 10))
        self.trail_length_max = _int_setting(settings, "trail_length_max", 80)
        self.trail_length_max = max(self.trail_length_min, min(self.trail_length_max, 400))
        self.trail_fade_power = _clamp(_float_setting(settings, "trail_fade_power", 1.5), 0.1, 10.0)
        requested_mode = settings.get("flow_mode", "perlin")
        self.flow_mode = requested_mode if requested_mode in FLOW_MODE_KEYS else "perlin"
        self.video_field_blend = 0.45  # how aggressively we adopt freshly detected directions
        
        # Particles that follow the flow field
        requested_particles = _int_setting(settings, "num_particles", 120)
        self.num_particles = max(MIN_PARTICLES, min(requested_particles, MAX_PARTICLES))
        self.particles = []
        
        for _ in range(self.num_particles):
            self.particles.append(self._create_particle())
    
    def _generate_particle_color(self):
        """Generate colors for particles"""
        color_themes = [
            # Deep blues and cyans
            [(20, 100, 200), (30, 150, 255), (60, 200, 255)],
            # Warm oranges and reds
            [(200, 80, 20), (255, 120, 40), (255, 160, 80)],
            # Purples and magentas
            [(120, 20, 200), (160, 60, 255), (200, 100, 255)],
            # Greens and teals
            [(20, 150, 100), (40, 200, 120), (80, 255, 160)],
            # Golden yellows
            [(180, 140, 20), (220, 180, 40), (255, 220, 80)]
        ]
        
        theme = random.choice(color_themes)
        return random.choice(theme)

    def _create_particle(self):
        return {
            'x': random.uniform(0, self.width),
            'y': random.uniform(0, self.height),
            'vx': 0.0,
            'vy': 0.0,
            'trail': [],
            'color': self._generate_particle_color(),
            'max_trail_length': random.randint(self.trail_length_min, self.trail_length_max),
            'speed_multiplier': random.uniform(3, 8),
            'age': 0
        }
    
    def update_flow_field(self, time_step, frame=None):
        """Update the flow field vectors using the selected generation method."""
        if self.flow_mode == "perlin" or frame is None:
            self.time_offset += time_step * self.time_speed

            for row in range(self.rows):
                for col in range(self.cols):
                    x = col * self.grid_size
                    y = row * self.grid_size

                    noise_x = x * self.noise_scale
                    noise_y = y * self.noise_scale
                    noise_t = self.time_offset * 0.1

                    noise_value = self.noise.octave_noise(
                        noise_x,
                        noise_y + noise_t,
                        octaves=3,
                        persistence=0.5,
                        scale=1.0
                    )

                    angle = (noise_value + 1.0) * math.pi

                    curl_noise = self.noise.octave_noise(
                        noise_x + 100,
                        noise_y + noise_t + 100,
                        octaves=2,
                        persistence=0.3,
                        scale=0.5
                    )

                    angle += curl_noise * self.curl_strength

                    self.flow_field[row, col, 0] = math.cos(angle) * self.force_magnitude
                    self.flow_field[row, col, 1] = math.sin(angle) * self.force_magnitude
            return

        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if self.flow_mode == "edge_filter":
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            processed = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(processed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        edges = cv2.Canny(processed, 80, 160)

        mode = self.flow_mode
        new_field = None

        if mode == "gradient":
            new_field = self._vector_field_from_gradients(grad_x, grad_y, magnitude)
        elif mode == "canny":
            edge_mask = (edges.astype(np.float32) / 255.0)
            new_field = self._vector_field_from_gradients(grad_x, grad_y, magnitude, mask=edge_mask)
        elif mode == "structure":
            new_field = self._vector_field_from_structure_tensor(grad_x, grad_y)
        elif mode == "hough":
            new_field = self._vector_field_from_hough(edges)
        elif mode == "edge_filter":
            new_field = self._vector_field_from_gradients(grad_x, grad_y, magnitude)
        else:
            new_field = self._vector_field_from_gradients(grad_x, grad_y, magnitude)

        if new_field is None:
            return

        if not np.any(new_field):
            self.flow_field *= 0.9
            return

        blend = np.clip(self.video_field_blend, 0.05, 1.0)
        self.flow_field = (1.0 - blend) * self.flow_field + blend * new_field
    
    def get_flow_at_position(self, x, y):
        """Get flow vector at a specific position using bilinear interpolation"""
        # Convert world coordinates to grid coordinates
        grid_x = x / self.grid_size
        grid_y = y / self.grid_size
        
        # Get integer grid positions
        x0 = int(grid_x)
        y0 = int(grid_y)
        x1 = min(x0 + 1, self.cols - 1)
        y1 = min(y0 + 1, self.rows - 1)
        
        # Ensure we're within bounds
        x0 = max(0, min(x0, self.cols - 1))
        y0 = max(0, min(y0, self.rows - 1))
        
        # Get fractional parts for interpolation
        fx = grid_x - x0
        fy = grid_y - y0
        
        # Bilinear interpolation of flow vectors
        if x0 < self.cols and y0 < self.rows:
            # Get the four surrounding flow vectors
            v00 = self.flow_field[y0, x0]
            v10 = self.flow_field[y0, x1] if x1 < self.cols else v00
            v01 = self.flow_field[y1, x0] if y1 < self.rows else v00
            v11 = self.flow_field[y1, x1] if (x1 < self.cols and y1 < self.rows) else v00
            
            # Interpolate
            v_top = v00 * (1 - fx) + v10 * fx
            v_bottom = v01 * (1 - fx) + v11 * fx
            flow = v_top * (1 - fy) + v_bottom * fy
            
            return flow[0], flow[1]
        
        return 0.0, 0.0
    
    def update_particles(self):
        """Update particle positions based on flow field"""
        for particle in self.particles:
            particle['age'] += 1
            
            # Get flow force at particle position
            flow_x, flow_y = self.get_flow_at_position(particle['x'], particle['y'])
            
            # Apply flow force to velocity (with some momentum)
            damping = 0.98  # Higher damping for smoother movement
            particle['vx'] = particle['vx'] * damping + flow_x * particle['speed_multiplier'] * self.speed_factor
            particle['vy'] = particle['vy'] * damping + flow_y * particle['speed_multiplier'] * self.speed_factor
            
            # Limit maximum velocity for smoother trails
            max_velocity = 200  # Reduced max velocity
            velocity_magnitude = math.sqrt(particle['vx']**2 + particle['vy']**2)
            if velocity_magnitude > max_velocity:
                particle['vx'] = (particle['vx'] / velocity_magnitude) * max_velocity
                particle['vy'] = (particle['vy'] / velocity_magnitude) * max_velocity
            
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Add current position to trail
            particle['trail'].append((int(particle['x']), int(particle['y'])))
            
            # Limit trail length
            if len(particle['trail']) > particle['max_trail_length']:
                particle['trail'].pop(0)
            
            # Reset particle if it goes off screen or gets too old
            if (particle['x'] < -50 or particle['x'] > self.width + 50 or 
                particle['y'] < -50 or particle['y'] > self.height + 50 or
                particle['age'] > 2000):
                self._reset_particle(particle)
    
    def _reset_particle(self, particle):
        """Reset a particle to a new random position"""
        edge = random.randint(0, 3)  # 0=top, 1=right, 2=bottom, 3=left
        
        if edge == 0:  # top
            particle['x'] = random.uniform(0, self.width)
            particle['y'] = random.uniform(-50, 0)
        elif edge == 1:  # right
            particle['x'] = random.uniform(self.width, self.width + 50)
            particle['y'] = random.uniform(0, self.height)
        elif edge == 2:  # bottom
            particle['x'] = random.uniform(0, self.width)
            particle['y'] = random.uniform(self.height, self.height + 50)
        else:  # left
            particle['x'] = random.uniform(-50, 0)
            particle['y'] = random.uniform(0, self.height)
        
        particle['vx'] = 0.0
        particle['vy'] = 0.0
        particle['trail'] = []
        particle['color'] = self._generate_particle_color()
        particle['max_trail_length'] = random.randint(self.trail_length_min, self.trail_length_max)  # Longer trails
        particle['speed_multiplier'] = random.uniform(5, 8) 
        particle['age'] = 0

    def _enforce_trail_length_limits(self):
        for particle in self.particles:
            if (particle['max_trail_length'] < self.trail_length_min or
                    particle['max_trail_length'] > self.trail_length_max):
                particle['max_trail_length'] = random.randint(self.trail_length_min, self.trail_length_max)
            if len(particle['trail']) > particle['max_trail_length']:
                particle['trail'] = particle['trail'][-particle['max_trail_length']:]

    def cycle_flow_mode(self, step=1):
        if self.flow_mode not in FLOW_MODE_KEYS:
            self.flow_mode = "perlin"
        idx = FLOW_MODE_KEYS.index(self.flow_mode)
        self.flow_mode = FLOW_MODE_KEYS[(idx + step) % len(FLOW_MODE_KEYS)]
        print(f"Flow mode set to {self.flow_mode_label()}")

    def flow_mode_label(self):
        return FLOW_MODE_LABELS.get(self.flow_mode, "Perlin Noise")

    def _aggregate_vector_field(self, vx, vy, weight):
        vx = vx.astype(np.float32)
        vy = vy.astype(np.float32)
        weight = weight.astype(np.float32)

        vx_weighted = vx * weight
        vy_weighted = vy * weight

        sum_weight = cv2.resize(weight, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        vx_sum = cv2.resize(vx_weighted, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        vy_sum = cv2.resize(vy_weighted, (self.cols, self.rows), interpolation=cv2.INTER_AREA)

        sum_weight = np.maximum(sum_weight, 1e-6)
        avg_vx = vx_sum / sum_weight
        avg_vy = vy_sum / sum_weight

        strength = sum_weight / (np.max(sum_weight) + 1e-6)
        strength = np.clip(strength, 0.0, 1.0)

        direction = np.stack((avg_vx, avg_vy), axis=-1)
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        direction = direction / norms

        mask = sum_weight > 1e-3
        direction[~mask] = 0.0
        strength[~mask] = 0.0

        return direction.astype(np.float32), strength.astype(np.float32)

    def _vector_field_from_gradients(self, grad_x, grad_y, magnitude, mask=None):
        if mask is None:
            weight = magnitude
        else:
            weight = magnitude * mask

        weight = np.maximum(weight, 0.0)

        tangent_x = -grad_y
        tangent_y = grad_x

        direction, strength = self._aggregate_vector_field(tangent_x, tangent_y, weight)
        return direction * (self.force_magnitude * strength[..., np.newaxis])

    def _vector_field_from_structure_tensor(self, grad_x, grad_y):
        grad_x = grad_x.astype(np.float32)
        grad_y = grad_y.astype(np.float32)
        ixx = grad_x * grad_x
        ixy = grad_x * grad_y
        iyy = grad_y * grad_y

        agg_xx = cv2.resize(ixx, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        agg_xy = cv2.resize(ixy, (self.cols, self.rows), interpolation=cv2.INTER_AREA)
        agg_yy = cv2.resize(iyy, (self.cols, self.rows), interpolation=cv2.INTER_AREA)

        new_field = np.zeros_like(self.flow_field, dtype=np.float32)
        strength = np.sqrt(agg_xx + agg_yy)
        strength_norm = strength / (np.max(strength) + 1e-6)

        for row in range(self.rows):
            for col in range(self.cols):
                tensor = np.array([
                    [agg_xx[row, col], agg_xy[row, col]],
                    [agg_xy[row, col], agg_yy[row, col]]
                ], dtype=np.float32)

                if np.allclose(tensor, 0.0):
                    continue

                eigenvalues, eigenvectors = np.linalg.eigh(tensor)
                dominant = eigenvectors[:, np.argmax(eigenvalues)]
                tangent = np.array([-dominant[1], dominant[0]], dtype=np.float32)
                norm = np.linalg.norm(tangent)
                if norm < 1e-6:
                    continue
                tangent /= norm
                new_field[row, col] = tangent * (self.force_magnitude * strength_norm[row, col])

        return new_field

    def _vector_field_from_hough(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=12)
        accum = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        counts = np.zeros((self.rows, self.cols), dtype=np.float32)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 1e-3:
                    continue
                direction = np.array([dx / length, dy / length], dtype=np.float32)
                samples = max(int(length / self.grid_size), 1)
                for idx in range(samples + 1):
                    t = idx / max(samples, 1)
                    sx = x1 + dx * t
                    sy = y1 + dy * t
                    col = int(sx / self.grid_size)
                    row = int(sy / self.grid_size)
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        accum[row, col] += direction
                        counts[row, col] += 1.0

        new_field = np.zeros_like(self.flow_field, dtype=np.float32)
        valid = counts > 0
        if np.any(valid):
            new_field[valid] = accum[valid] / counts[valid][..., np.newaxis]
            norms = np.linalg.norm(new_field, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            new_field = new_field / norms

        strength = counts / (np.max(counts) + 1e-6)
        strength = np.clip(strength, 0.0, 1.0)
        new_field *= (self.force_magnitude * strength[..., np.newaxis])

        return new_field

    def adjust_trail_length_min(self, delta):
        max_cap = 400
        self.trail_length_min = max(1, self.trail_length_min + delta)
        self.trail_length_min = min(self.trail_length_min, max_cap)
        if self.trail_length_min > self.trail_length_max:
            self.trail_length_max = self.trail_length_min
        self._enforce_trail_length_limits()

    def adjust_trail_length_max(self, delta):
        max_cap = 400
        self.trail_length_max = max(self.trail_length_min, min(self.trail_length_max + delta, max_cap))
        self._enforce_trail_length_limits()

    def adjust_trail_length_range(self, delta):
        max_cap = 400
        new_min = max(1, min(self.trail_length_min + delta, max_cap))
        new_max = max(new_min, min(self.trail_length_max + delta, max_cap))
        self.trail_length_min = int(new_min)
        self.trail_length_max = int(new_max)
        self._enforce_trail_length_limits()

    def adjust_trail_fade_power(self, delta):
        self.trail_fade_power = _clamp(self.trail_fade_power + delta, 0.1, 10.0)

    def adjust_particle_count(self, delta):
        new_target = max(MIN_PARTICLES, min(self.num_particles + delta, MAX_PARTICLES))
        if new_target == self.num_particles:
            return
        if new_target > self.num_particles:
            for _ in range(new_target - self.num_particles):
                self.particles.append(self._create_particle())
        else:
            self.particles = self.particles[:new_target]
        self.num_particles = new_target
        self._enforce_trail_length_limits()

    def to_settings(self):
        """Return adjustable parameters for persistence."""
        return {
            "noise_scale": self.noise_scale,
            "time_speed": self.time_speed,
            "speed_factor": self.speed_factor,
            "base_trail_thickness": self.base_trail_thickness,
            "force_magnitude": self.force_magnitude,
            "curl_strength": self.curl_strength,
            "trail_length_min": self.trail_length_min,
            "trail_length_max": self.trail_length_max,
            "trail_fade_power": self.trail_fade_power,
            "num_particles": self.num_particles,
            "flow_mode": self.flow_mode,
        }
    
    def draw_flow_field(self, canvas):
        """Render flow vectors onto the provided canvas."""
        for row in range(0, self.rows, 2):  # Skip some for clarity
            for col in range(0, self.cols, 2):
                x = col * self.grid_size
                y = row * self.grid_size

                if x < self.width and y < self.height:
                    flow_x = self.flow_field[row, col, 0] * 30
                    flow_y = self.flow_field[row, col, 1] * 30

                    end_x = int(x + flow_x)
                    end_y = int(y + flow_y)

                    cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (100, 100, 100), 1, cv2.LINE_AA)
    
    def draw_particles(self, canvas):
        """Draw particles and their trails"""
        for particle in self.particles:
            trail = particle['trail']
            if len(trail) < 2:
                continue
            
            color = particle['color']
            
            # Draw trail with fading effect and anti-aliasing
            for i in range(len(trail) - 1):
                # Calculate alpha based on position in trail (newer points brighter)
                progress = i / len(trail) if len(trail) > 1 else 1.0
                progress = _clamp(progress, 0.0, 1.0)
                fade = progress ** self.trail_fade_power
                alpha = _clamp(fade * 0.9 + 0.05, 0.0, 1.0)
                
                # Apply alpha to color
                faded_color = tuple(int(c * alpha) for c in color)
                
                # Draw trail segment with anti-aliasing
                thickness = max(1, int(self.base_trail_thickness * alpha))
                
                # Use anti-aliased line drawing for smoother appearance
                cv2.line(canvas, trail[i], trail[i + 1], faded_color, thickness, cv2.LINE_AA)
            
            # No particle head circles - only flowing lines
    
    def draw(self, canvas, show_flow_vectors=False):
        """Draw the complete flow field visualization and return optional vector layer."""
        # Create fade effect
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(canvas, 0.97, overlay, 0.03, 0, canvas)
        
        vector_layer = None
        if show_flow_vectors:
            vector_layer = np.zeros_like(canvas)
            self.draw_flow_field(vector_layer)
        
        # Draw particles
        self.draw_particles(canvas)

        return vector_layer


def build_overlay_lines(flow_field, show_vectors, show_controls, show_video):
    if not show_controls:
        return []

    lines = [
        "Flow Field Controls (?: toggle)",
        f"Vectors: {'ON' if show_vectors else 'OFF'} (v)",
        f"Video feed: {'ON' if show_video else 'OFF'} (b)",
        f"Flow mode: {flow_field.flow_mode_label()} (m)",
        "SPACE: reset scene | r: reset particles | n: new noise",
        f"Noise scale: {flow_field.noise_scale:.4f} (1/2)",
        f"Time speed: {flow_field.time_speed:.3f} (3/4)",
        f"Line speed: {flow_field.speed_factor:.3f} (5/6)",
    f"Particles: {flow_field.num_particles} (u/i)",
        f"Trail thickness: {flow_field.base_trail_thickness:.2f} (7/8)",
    f"Trail length range: {flow_field.trail_length_min}-{flow_field.trail_length_max} (f/g min, h/j max, z/x both)",
    f"Trail fade power: {flow_field.trail_fade_power:.2f} (k faster, l slower)",
        f"Flow strength: {flow_field.force_magnitude:.3f} (9/0)",
        f"Curl strength: {flow_field.curl_strength:.2f} ([/])",
        "Save controls to file: (p)",
        "ESC: quit",
    ]
    return lines


def draw_overlay(frame, lines):
    if not lines:
        return frame

    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    padding_x = 10
    padding_y = 8
    line_gap = 4

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_height = max((size[1] for size in sizes), default=0)
    panel_width = max((size[0] for size in sizes), default=0) + padding_x * 2
    panel_height = len(lines) * (text_height + line_gap) - line_gap + padding_y * 2 if lines else 0

    top_left = (8, 8)
    bottom_right = (8 + panel_width, 8 + panel_height)

    overlay = output.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)

    for idx, (line, size) in enumerate(zip(lines, sizes)):
        x = top_left[0] + padding_x
        y = top_left[1] + padding_y + idx * (text_height + line_gap) + size[1]
        cv2.putText(output, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return output


def main():
    """Main loop for the flow field test"""
    default_width, default_height = 800, 600
    
    loaded_settings = load_settings()

    cap = cv2.VideoCapture(0)
    initial_frame = None

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            initial_frame = frame
        else:
            cap.release()
            cap = None
            width, height = default_width, default_height
    else:
        cap = None
        width, height = default_width, default_height

    # Initialize the flow field
    flow_field = FlowField(width, height, settings=loaded_settings)
    
    # Create window
    cv2.namedWindow("Flow Field Art", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Flow Field Art", width, height)
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    show_vectors = bool(loaded_settings.get("show_vectors", False))
    show_controls = bool(loaded_settings.get("show_controls", True))
    show_video = bool(loaded_settings.get("show_video", cap is not None))
    clock = time.time()

    frame = initial_frame if initial_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - clock
            clock = current_time
            
            if cap is not None:
                ret, current_frame = cap.read()
                if ret:
                    if current_frame.shape[1] != width or current_frame.shape[0] != height:
                        frame = cv2.resize(current_frame, (width, height))
                    else:
                        frame = current_frame
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Update the flow field
            flow_field.update_flow_field(dt, frame)
            
            # Update particles
            flow_field.update_particles()
            
            vector_layer = flow_field.draw(canvas, show_vectors)

            flow_frame = canvas.copy()

            if show_video:
                base_frame = frame.copy()
                display_frame = cv2.add(base_frame, flow_frame)
            else:
                display_frame = flow_frame

            if vector_layer is not None:
                display_frame = cv2.add(display_frame, vector_layer)

            display_frame = draw_overlay(display_frame, build_overlay_lines(flow_field, show_vectors, show_controls, show_video))
            cv2.imshow("Flow Field Art", display_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - reset
                current_settings = flow_field.to_settings()
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                flow_field = FlowField(width, height, settings=current_settings)
            elif key == ord('v'):  # V - toggle vectors
                show_vectors = not show_vectors
            elif key == ord('b'):  # B - toggle video feed
                show_video = not show_video
            elif key == ord('m'):
                flow_field.cycle_flow_mode()
            elif key == ord('r'):  # R - regenerate particles
                for particle in flow_field.particles:
                    flow_field._reset_particle(particle)
            elif key == ord('n'):  # N - new noise field
                flow_field.noise = PerlinNoise()  # Generate new random seed
            elif key == ord('1'):
                flow_field.noise_scale = max(0.001, flow_field.noise_scale * 0.9)
            elif key == ord('2'):
                flow_field.noise_scale = min(0.1, flow_field.noise_scale * 1.1)
            elif key == ord('3'):
                flow_field.time_speed = max(0.01, flow_field.time_speed * 0.9)
            elif key == ord('4'):
                flow_field.time_speed = min(5.0, flow_field.time_speed * 1.1)
            elif key == ord('5'):
                flow_field.speed_factor = max(0.1, flow_field.speed_factor * 0.9)
            elif key == ord('6'):
                flow_field.speed_factor = min(2.0, flow_field.speed_factor * 1.1)
            elif key == ord('u'):
                flow_field.adjust_particle_count(-10)
            elif key == ord('i'):
                flow_field.adjust_particle_count(10)
            elif key == ord('7'):
                flow_field.base_trail_thickness = max(0.5, flow_field.base_trail_thickness - 0.5)
            elif key == ord('8'):
                flow_field.base_trail_thickness = min(15.0, flow_field.base_trail_thickness + 0.5)
            elif key == ord('k'):
                flow_field.adjust_trail_fade_power(0.2)
            elif key == ord('l'):
                flow_field.adjust_trail_fade_power(-0.2)
            elif key == ord('f'):
                flow_field.adjust_trail_length_min(-5)
            elif key == ord('g'):
                flow_field.adjust_trail_length_min(5)
            elif key == ord('h'):
                flow_field.adjust_trail_length_max(-5)
            elif key == ord('j'):
                flow_field.adjust_trail_length_max(5)
            elif key == ord('z'):
                flow_field.adjust_trail_length_range(-10)
            elif key == ord('x'):
                flow_field.adjust_trail_length_range(10)
            elif key == ord('9'):
                flow_field.force_magnitude = max(0.05, flow_field.force_magnitude * 0.9)
            elif key == ord('0'):
                flow_field.force_magnitude = min(2.0, flow_field.force_magnitude * 1.1)
            elif key == ord('['):
                flow_field.curl_strength = max(0.0, flow_field.curl_strength - 0.1)
            elif key == ord(']'):
                flow_field.curl_strength = min(2.0, flow_field.curl_strength + 0.1)
            elif key == ord('?'):
                show_controls = not show_controls
            elif key == ord('p'):
                save_settings(flow_field, show_vectors, show_controls, show_video)
            
            # Control frame rate
            time.sleep(0.016)  # ~60 FPS
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()