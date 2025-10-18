import cv2
import numpy as np
import math
import time
import random

class FlowingBrushStrokes:
    """
    Creates continuous flowing brush strokes that move outward like organic curved lines.
    Each stroke has its own color, speed, length, and smooth curved path.
    """
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Create more flow lines with different properties
        self.num_lines = 35  # More lines for richer effect
        self.flow_lines = []
        
        for i in range(self.num_lines):
            # Each line has its own properties
            base_angle = (2 * math.pi * i) / self.num_lines
            angle_variation = random.uniform(-0.4, 0.4)
            start_angle = base_angle + angle_variation
            
            speed = random.uniform(0.8, 3.5)  # pixels per frame
            color = self._generate_flow_color()
            thickness = random.randint(1, 4)
            max_length = random.randint(150, 400)  # Varying line lengths
            
            # Curve properties for smooth organic movement
            curve_frequency = random.uniform(0.01, 0.05)  # How wavy the line is
            curve_amplitude = random.uniform(10, 40)      # How much it curves
            
            self.flow_lines.append({
                'start_angle': start_angle,
                'current_angle': start_angle,
                'speed': speed,
                'color': color,
                'thickness': thickness,
                'max_length': max_length,
                'current_length': 0.0,
                'curve_frequency': curve_frequency,
                'curve_amplitude': curve_amplitude,
                'points': [],  # Store the continuous path
                'age': 0,
                'growing': True,
                'fade_start': random.randint(60, 120)  # When to start fading
            })
    
    def _generate_flow_color(self):
        """Generate vibrant flowing colors"""
        color_themes = [
            # Electric blues and cyans
            [(0, 150, 255), (0, 200, 255), (50, 255, 255)],
            # Warm oranges and reds  
            [(255, 100, 0), (255, 150, 50), (255, 200, 100)],
            # Purples and magentas
            [(150, 0, 255), (200, 50, 255), (255, 100, 200)],
            # Greens and teals
            [(0, 255, 100), (50, 255, 150), (100, 255, 200)],
            # Golden yellows
            [(255, 200, 0), (255, 225, 50), (255, 255, 100)]
        ]
        
        theme = random.choice(color_themes)
        return random.choice(theme)
    
    def update(self, frame_time):
        """Update the continuous flow animation"""
        for line in self.flow_lines:
            line['age'] += 1
            
            if line['growing'] and line['current_length'] < line['max_length']:
                # Grow the line by adding new points
                line['current_length'] += line['speed']
                
                # Calculate the current curve offset
                curve_offset = math.sin(line['current_length'] * line['curve_frequency']) * line['curve_amplitude']
                current_angle = line['start_angle'] + curve_offset * 0.01
                
                # Calculate new point position
                x = self.center_x + math.cos(current_angle) * line['current_length']
                y = self.center_y + math.sin(current_angle) * line['current_length']
                
                # Keep points within bounds (with some padding for smooth exit)
                if -100 <= x <= self.width + 100 and -100 <= y <= self.height + 100:
                    line['points'].append((int(x), int(y)))
                else:
                    line['growing'] = False
            
            elif not line['growing'] or line['current_length'] >= line['max_length']:
                # Line finished growing, start aging/fading
                line['growing'] = False
                
                # Remove oldest points to create flowing effect
                if len(line['points']) > 0 and line['age'] > line['fade_start']:
                    # Remove points from the beginning to create trailing effect
                    points_to_remove = max(1, int(line['speed'] * 0.5))
                    line['points'] = line['points'][points_to_remove:]
                
                # Reset line when it's fully faded
                if len(line['points']) == 0:
                    self._reset_line(line)
    
    def _reset_line(self, line):
        """Reset a line with new random properties"""
        base_angle = random.uniform(0, 2 * math.pi)
        line['start_angle'] = base_angle
        line['current_angle'] = base_angle
        line['speed'] = random.uniform(0.8, 3.5)
        line['color'] = self._generate_flow_color()
        line['thickness'] = random.randint(1, 4)
        line['max_length'] = random.randint(150, 400)
        line['current_length'] = 0.0
        line['curve_frequency'] = random.uniform(0.01, 0.05)
        line['curve_amplitude'] = random.uniform(10, 40)
        line['points'] = []
        line['age'] = 0
        line['growing'] = True
        line['fade_start'] = random.randint(60, 120)
    
    def draw(self, canvas):
        """Draw the continuous flowing brush strokes on the canvas"""
        # Create a subtle fade effect for trails
        # overlay = canvas.copy()
        # cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        # cv2.addWeighted(canvas, 0.95, overlay, 0.05, 0, canvas)
        
        for line in self.flow_lines:
            points = line['points']
            if len(points) < 2:
                continue
                
            color = line['color']
            thickness = line['thickness']
            
            # Draw the continuous curved line with varying opacity
            for i in range(len(points) - 1):
                # Calculate alpha based on position in the line (newer points brighter)
                progress = i / len(points) if len(points) > 1 else 1.0
                alpha = progress * 0.3 + 0.7  # Keep most of the line visible
                # alpha = 1.0
                
                # Apply alpha to color
                faded_color = tuple(int(c * alpha) for c in color)
                
                # Draw line segment with smooth connection
                cv2.line(canvas, points[i], points[i + 1], faded_color, thickness)
                
                # Add subtle glow for thicker lines
                # if thickness > 2:
                #     glow_color = tuple(int(c * alpha * 0.6) for c in color)
                #     cv2.line(canvas, points[i], points[i + 1], glow_color, thickness + 2)
            
            # Draw the "head" of the growing line with extra brightness
            if line['growing'] and len(points) > 0:
                head_point = points[-1]
                bright_color = tuple(min(255, int(c * 1.3)) for c in color)
                # cv2.circle(canvas, head_point, thickness , bright_color, -1)
                
                # Add a small glow around the head
                glow_color = tuple(int(c * 0.8) for c in color)
                cv2.circle(canvas, head_point, thickness + 3, glow_color, 1)


def main():
    """Main loop for the generative art test"""
    width, height = 800, 600
    
    # Initialize the flowing brush strokes
    flow_art = FlowingBrushStrokes(width, height)
    
    # Create window
    cv2.namedWindow("Generative Flow Art", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Generative Flow Art", width, height)
    
    # Main canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    print("Generative Flow Art Test")
    print("Press ESC to exit")
    print("Press SPACE to reset")
    print("Press 'r' to randomize colors")
    
    clock = time.time()
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - clock
            clock = current_time
            
            # Update the animation
            flow_art.update(dt)
            
            # Draw the art
            flow_art.draw(canvas)
            
            # Show the result
            cv2.imshow("Generative Flow Art", canvas)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - reset
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                flow_art = FlowingBrushStrokes(width, height)
            elif key == ord('r'):  # R - randomize colors
                # Randomize colors and reset some lines
                for line in flow_art.flow_lines:
                    line['color'] = flow_art._generate_flow_color()
                    # Reset some lines for immediate visual change
                    if random.random() < 0.3:  # 30% chance to reset each line
                        flow_art._reset_line(line)
            
            # Control frame rate
            time.sleep(0.016)  # ~60 FPS
    
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()