import cv2
import numpy as np
import math
import time
import random

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
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
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
        self.noise_scale = 0.01  # Smaller values => smoother field variation
        self.time_speed = 0.5    # Controls how quickly the flow evolves over time
        self.force_magnitude = 0.4  # Base strength applied from the flow field
        self.curl_strength = 0.5  # Additional curl applied for swirling motion
        self.speed_factor = 0.5   # Global multiplier for particle speed
        self.base_trail_thickness = 3.0  # Controls visual thickness of trails
        
        # Particles that follow the flow field
        self.num_particles = 120  # More particles for denser, smoother trails
        self.particles = []
        
        for i in range(self.num_particles):
            self.particles.append({
                'x': random.uniform(0, width),
                'y': random.uniform(0, height),
                'vx': 0.0,
                'vy': 0.0,
                'trail': [],  # Store particle trail
                'color': self._generate_particle_color(),
                'max_trail_length': random.randint(10, 80),  # Longer trails for smoother appearance
                'speed_multiplier': random.uniform(3, 8),  # Slower for smoother movement
                'age': 0
            })
    
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
    
    def update_flow_field(self, time_step):
        """Update the flow field vectors using Perlin noise"""
        self.time_offset += time_step * self.time_speed
        
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate world position for this grid cell
                x = col * self.grid_size
                y = row * self.grid_size
                
                # Generate Perlin noise at this position and time
                # Use octave noise for more interesting patterns
                noise_x = x * self.noise_scale
                noise_y = y * self.noise_scale
                noise_t = self.time_offset * 0.1
                
                # Get noise value and convert to angle (0 to 2π)
                noise_value = self.noise.octave_noise(
                    noise_x, 
                    noise_y + noise_t,  # Add time offset to Y for temporal evolution
                    octaves=3,
                    persistence=0.5,
                    scale=1.0
                )
                
                # Convert noise to angle (scale from [-1,1] to [0,2π])
                angle = (noise_value + 1.0) * math.pi
                
                # Add some curl/rotation for more interesting flow
                curl_noise = self.noise.octave_noise(
                    noise_x + 100,  # Offset to get different noise
                    noise_y + noise_t + 100,
                    octaves=2,
                    persistence=0.3,
                    scale=0.5
                )
                
                # Add curl to the angle
                angle += curl_noise * self.curl_strength
                
                # Convert to flow vector
                self.flow_field[row, col, 0] = math.cos(angle) * self.force_magnitude
                self.flow_field[row, col, 1] = math.sin(angle) * self.force_magnitude
    
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
        particle['max_trail_length'] = random.randint(10, 80)  # Longer trails
        particle['speed_multiplier'] = random.uniform(5, 8) 
        particle['age'] = 0
    
    def draw_flow_field(self, canvas, show_vectors=False):
        """Draw the flow field visualization"""
        if show_vectors:
            # Draw flow field vectors (for debugging)
            for row in range(0, self.rows, 2):  # Skip some for clarity
                for col in range(0, self.cols, 2):
                    x = col * self.grid_size
                    y = row * self.grid_size
                    
                    if x < self.width and y < self.height:
                        flow_x = self.flow_field[row, col, 0] * 30
                        flow_y = self.flow_field[row, col, 1] * 30
                        
                        end_x = int(x + flow_x)
                        end_y = int(y + flow_y)
                        
                        # Use anti-aliasing for flow vector arrows too
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
                alpha = progress * 0.5 + 0.3
                
                # Apply alpha to color
                faded_color = tuple(int(c * alpha) for c in color)
                
                # Draw trail segment with anti-aliasing
                thickness = max(1, int(self.base_trail_thickness * alpha))
                
                # Use anti-aliased line drawing for smoother appearance
                cv2.line(canvas, trail[i], trail[i + 1], faded_color, thickness, cv2.LINE_AA)
            
            # No particle head circles - only flowing lines
    
    def draw(self, canvas, show_flow_vectors=False):
        """Draw the complete flow field visualization"""
        # Create fade effect
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(canvas, 0.97, overlay, 0.03, 0, canvas)
        
        # Draw flow field vectors (optional)
        if show_flow_vectors:
            self.draw_flow_field(canvas, True)
        
        # Draw particles
        self.draw_particles(canvas)


def main():
    """Main loop for the flow field test"""
    width, height = 800, 600
    
    # Initialize the flow field
    flow_field = FlowField(width, height)
    
    # Create window
    cv2.namedWindow("Flow Field Art", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Flow Field Art", width, height)
    
    # Main canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    print("Flow Field Generative Art - Perlin Noise Edition")
    print("Press ESC to exit")
    print("Press SPACE to reset")
    print("Press 'v' to toggle flow vectors")
    print("Press 'r' to regenerate particles")
    print("Press 'n' to regenerate noise field")
    print("Press '1/2' to adjust noise scale (field detail)")
    print("Press '3/4' to adjust flow evolution speed")
    print("Press '5/6' to adjust line speed")
    print("Press '7/8' to adjust line thickness")
    print("Press '9/0' to adjust flow strength")
    print("Press '[/]' to adjust curl strength")
    
    show_vectors = False
    clock = time.time()
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - clock
            clock = current_time
            
            # Update the flow field
            flow_field.update_flow_field(dt)
            
            # Update particles
            flow_field.update_particles()
            
            # Draw everything
            flow_field.draw(canvas, show_vectors)
            
            # Show the result
            cv2.imshow("Flow Field Art", canvas)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - reset
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                flow_field = FlowField(width, height)
            elif key == ord('v'):  # V - toggle vectors
                show_vectors = not show_vectors
                print(f"Flow vectors: {'ON' if show_vectors else 'OFF'}")
            elif key == ord('r'):  # R - regenerate particles
                for particle in flow_field.particles:
                    flow_field._reset_particle(particle)
                print("Particles regenerated")
            elif key == ord('n'):  # N - new noise field
                flow_field.noise = PerlinNoise()  # Generate new random seed
                print("Noise field regenerated")
            elif key == ord('1'):
                flow_field.noise_scale = max(0.001, flow_field.noise_scale * 0.9)
                print(f"Noise scale: {flow_field.noise_scale:.4f}")
            elif key == ord('2'):
                flow_field.noise_scale = min(0.1, flow_field.noise_scale * 1.1)
                print(f"Noise scale: {flow_field.noise_scale:.4f}")
            elif key == ord('3'):
                flow_field.time_speed = max(0.01, flow_field.time_speed * 0.9)
                print(f"Flow evolution speed: {flow_field.time_speed:.3f}")
            elif key == ord('4'):
                flow_field.time_speed = min(5.0, flow_field.time_speed * 1.1)
                print(f"Flow evolution speed: {flow_field.time_speed:.3f}")
            elif key == ord('5'):
                flow_field.speed_factor = max(0.1, flow_field.speed_factor * 0.9)
                print(f"Line speed factor: {flow_field.speed_factor:.3f}")
            elif key == ord('6'):
                flow_field.speed_factor = min(2.0, flow_field.speed_factor * 1.1)
                print(f"Line speed factor: {flow_field.speed_factor:.3f}")
            elif key == ord('7'):
                flow_field.base_trail_thickness = max(1.0, flow_field.base_trail_thickness - 0.5)
                print(f"Line thickness: {flow_field.base_trail_thickness:.2f}")
            elif key == ord('8'):
                flow_field.base_trail_thickness = min(10.0, flow_field.base_trail_thickness + 0.5)
                print(f"Line thickness: {flow_field.base_trail_thickness:.2f}")
            elif key == ord('9'):
                flow_field.force_magnitude = max(0.05, flow_field.force_magnitude * 0.9)
                print(f"Flow strength: {flow_field.force_magnitude:.3f}")
            elif key == ord('0'):
                flow_field.force_magnitude = min(2.0, flow_field.force_magnitude * 1.1)
                print(f"Flow strength: {flow_field.force_magnitude:.3f}")
            elif key == ord('['):
                flow_field.curl_strength = max(0.0, flow_field.curl_strength - 0.1)
                print(f"Curl strength: {flow_field.curl_strength:.2f}")
            elif key == ord(']'):
                flow_field.curl_strength = min(2.0, flow_field.curl_strength + 0.1)
                print(f"Curl strength: {flow_field.curl_strength:.2f}")
            
            # Control frame rate
            time.sleep(0.016)  # ~60 FPS
    
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()