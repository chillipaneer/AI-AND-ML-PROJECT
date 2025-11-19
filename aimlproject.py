import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import string

class EnhancedAirDrawingSystem:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Drawing configuration with gradient colors
        self.colors = {
            'red': (0, 0, 255),
            'orange': (0, 140, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'blue': (255, 0, 0),
            'purple': (255, 0, 127),
            'magenta': (255, 0, 255),
            'pink': (203, 192, 255),
            'white': (255, 255, 255)
        }
        self.color_names = list(self.colors.keys())
        self.current_color_index = 0
        self.current_color = self.colors['red']
        self.brush_size = 5
        
        # Drawing state
        self.drawing_canvas = None
        self.ui_canvas = None
        self.prev_x, self.prev_y = None, None
        self.gesture_mode = 'idle'
        self.last_gesture_change = 0
        self.last_color_change = 0
        
        # Point buffer for smoothing
        self.points_buffer = deque(maxlen=5)
        
        # Sign language recognition
        self.sign_language_letters = {
            'A': [1, 0, 0, 0, 0],  # Thumb up, all others down
            'B': [0, 1, 1, 1, 1],  # All fingers up except thumb
            'C': [1, 1, 1, 1, 0],  # Curved hand (approximation)
            'D': [0, 1, 0, 0, 0],  # Only index finger up
            'F': [0, 0, 1, 1, 1],  # Three fingers up (middle, ring, pinky)
            'I': [0, 0, 0, 0, 1],  # Only pinky up
            'L': [1, 1, 0, 0, 0],  # Thumb and index up
            'V': [0, 1, 1, 0, 0],  # Peace sign
            'W': [0, 1, 1, 1, 0],  # Three fingers up (index, middle, ring)
            'Y': [1, 0, 0, 0, 1],  # Thumb and pinky up
        }
        
        self.detected_letter = None
        self.letter_confidence = 0
        self.letter_history = deque(maxlen=15)
        self.recognized_text = ""
        self.last_letter_time = 0
        
        # Feature toggles
        self.show_sign_language = True
        self.show_drawing_trail = True
        self.rainbow_mode = False
        self.glow_effect = True
        
        # Animation variables
        self.ui_alpha = 0.85
        self.particle_effects = []
        
    def recognize_gesture(self, landmarks):
        """Enhanced gesture recognition with sign language detection"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [3, 6, 10, 14, 18]
        
        fingers_up = []
        
        # Check thumb
        if landmarks[finger_tips[0]].x < landmarks[finger_bases[0]].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Check other fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_bases[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Sign language detection
        if self.show_sign_language:
            self.detect_sign_language(fingers_up)
        
        # Gesture recognition
        total_fingers = sum(fingers_up)
        
        if fingers_up == [0, 1, 0, 0, 0]:  # Only index
            return 'draw'
        elif fingers_up == [0, 1, 1, 0, 0]:  # Index and middle
            return 'select'
        elif total_fingers == 0:  # Fist
            return 'erase'
        elif fingers_up == [1, 1, 1, 1, 1]:  # All fingers - clear canvas
            return 'clear'
        elif total_fingers >= 3 and fingers_up[0] == 0:  # Three+ fingers, no thumb
            return 'color_change'
        
        return 'idle'
    
    def detect_sign_language(self, fingers_up):
        """Detect sign language letters"""
        best_match = None
        best_score = 0
        
        for letter, pattern in self.sign_language_letters.items():
            # Calculate similarity
            matches = sum([1 for i in range(5) if fingers_up[i] == pattern[i]])
            score = matches / 5.0
            
            if score > best_score and score >= 0.8:  # 80% match threshold
                best_score = score
                best_match = letter
        
        if best_match:
            self.letter_history.append(best_match)
            
            # Check for consistent detection
            if len(self.letter_history) >= 10:
                most_common = max(set(self.letter_history), key=self.letter_history.count)
                count = self.letter_history.count(most_common)
                
                if count >= 8:  # 80% consistency
                    self.detected_letter = most_common
                    self.letter_confidence = count / len(self.letter_history)
                    
                    # Add to recognized text if held for 2 seconds
                    current_time = time.time()
                    if current_time - self.last_letter_time > 2.0:
                        if self.detected_letter not in self.recognized_text[-1:]:
                            self.recognized_text += self.detected_letter
                            self.last_letter_time = current_time
                            self.add_particle_effect()
                            print(f"Recognized: {self.detected_letter} -> Text: {self.recognized_text}")
        else:
            self.detected_letter = None
            self.letter_confidence = 0
    
    def add_particle_effect(self):
        """Add particle effect for letter recognition"""
        for _ in range(10):
            self.particle_effects.append({
                'x': np.random.randint(50, 250),
                'y': np.random.randint(100, 200),
                'vx': np.random.uniform(-3, 3),
                'vy': np.random.uniform(-5, -1),
                'life': 30,
                'color': self.current_color
            })
    
    def update_particles(self, frame):
        """Update and draw particle effects"""
        for particle in self.particle_effects[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.particle_effects.remove(particle)
            else:
                alpha = particle['life'] / 30.0
                size = int(5 * alpha)
                cv2.circle(frame, (int(particle['x']), int(particle['y'])), 
                          size, particle['color'], -1)
    
    def get_smoothed_position(self, x, y):
        """Smooth the drawing position"""
        self.points_buffer.append((x, y))
        if len(self.points_buffer) > 0:
            avg_x = int(np.mean([p[0] for p in self.points_buffer]))
            avg_y = int(np.mean([p[1] for p in self.points_buffer]))
            return avg_x, avg_y
        return x, y
    
    def draw_line(self, x, y):
        """Draw a line with optional effects"""
        if self.prev_x is not None and self.prev_y is not None:
            color = self.current_color
            
            # Rainbow mode
            if self.rainbow_mode:
                hue = int((time.time() * 50) % 180)
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(map(int, color_bgr))
            
            # Draw main line
            cv2.line(self.drawing_canvas, (self.prev_x, self.prev_y), 
                    (x, y), color, self.brush_size)
            
            # Glow effect
            if self.glow_effect:
                cv2.line(self.drawing_canvas, (self.prev_x, self.prev_y), 
                        (x, y), color, self.brush_size + 4, cv2.LINE_AA)
                
        self.prev_x, self.prev_y = x, y
    
    def erase(self, x, y):
        """Enhanced erase with smooth circle"""
        cv2.circle(self.drawing_canvas, (x, y), 30, (0, 0, 0), -1)
        self.prev_x, self.prev_y = x, y
    
    def cycle_color(self):
        """Cycle to next color"""
        current_time = time.time()
        if current_time - self.last_color_change > 0.5:
            self.current_color_index = (self.current_color_index + 1) % len(self.color_names)
            color_name = self.color_names[self.current_color_index]
            self.current_color = self.colors[color_name]
            self.last_color_change = current_time
            print(f"Color changed to: {color_name}")
    
    def draw_modern_ui(self, frame):
        """Draw modern, attractive UI"""
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Draw semi-transparent top bar
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        cv2.addWeighted(overlay, self.ui_alpha, frame, 1 - self.ui_alpha, 0, frame)
        
        # Draw color palette with modern design
        palette_y = 50
        palette_start_x = 50
        for i, (name, color) in enumerate(self.colors.items()):
            x = palette_start_x + i * 60
            
            # Shadow effect
            cv2.circle(frame, (x+2, palette_y+2), 22, (0, 0, 0), -1)
            cv2.circle(frame, (x, palette_y), 22, color, -1)
            
            # Selection indicator with glow
            if i == self.current_color_index:
                cv2.circle(frame, (x, palette_y), 26, (255, 255, 255), 3)
                cv2.circle(frame, (x, palette_y), 30, color, 2)
        
        # Mode indicator with modern styling
        mode_colors = {
            'draw': (100, 255, 100),
            'select': (255, 255, 100),
            'erase': (100, 100, 255),
            'color_change': (255, 100, 255),
            'clear': (255, 150, 50),
            'idle': (150, 150, 150)
        }
        mode_color = mode_colors.get(self.gesture_mode, (255, 255, 255))
        
        # Draw mode panel
        cv2.rectangle(frame, (w - 280, 10), (w - 10, 90), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - 280, 10), (w - 10, 90), mode_color, 3)
        
        mode_text = f"MODE: {self.gesture_mode.upper()}"
        cv2.putText(frame, mode_text, (w - 260, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        brush_text = f"Brush: {self.brush_size}px"
        cv2.putText(frame, brush_text, (w - 260, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Sign language detection panel
        if self.show_sign_language:
            panel_height = 180 if self.recognized_text else 140
            cv2.rectangle(frame, (10, h - panel_height - 10), (400, h - 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h - panel_height - 10), (400, h - 10), (100, 200, 255), 2)
            
            cv2.putText(frame, "SIGN LANGUAGE", (25, h - panel_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            if self.detected_letter:
                # Large letter display
                cv2.putText(frame, self.detected_letter, (180, h - panel_height + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 100), 5)
                
                # Confidence bar
                confidence_width = int(self.letter_confidence * 200)
                cv2.rectangle(frame, (25, h - panel_height + 110), (225, h - panel_height + 125), 
                             (60, 60, 60), -1)
                cv2.rectangle(frame, (25, h - panel_height + 110), (25 + confidence_width, h - panel_height + 125), 
                             (100, 255, 100), -1)
            else:
                cv2.putText(frame, "Show hand sign...", (25, h - panel_height + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Recognized text
            if self.recognized_text:
                cv2.putText(frame, f"Text: {self.recognized_text}", (25, h - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        
        # Instructions panel
        instructions_x = 420
        cv2.rectangle(frame, (instructions_x, h - 140), (instructions_x + 300, h - 10), 
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (instructions_x, h - 140), (instructions_x + 300, h - 10), 
                     (200, 100, 200), 2)
        
        cv2.putText(frame, "GESTURES", (instructions_x + 15, h - 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 200), 2)
        
        instructions = [
            "1 finger: Draw",
            "2 fingers: Move cursor",
            "Fist: Erase",
            "3+ fingers: Change color"
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (instructions_x + 15, h - 85 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Keyboard shortcuts
        shortcuts_x = 740
        cv2.rectangle(frame, (shortcuts_x, h - 140), (shortcuts_x + 300, h - 10), 
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (shortcuts_x, h - 140), (shortcuts_x + 300, h - 10), 
                     (255, 200, 100), 2)
        
        cv2.putText(frame, "SHORTCUTS", (shortcuts_x + 15, h - 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        
        shortcuts = [
            "C: Clear canvas",
            "S: Save drawing",
            "+/-: Brush size",
            "R: Rainbow mode | Q: Quit"
        ]
        for i, shortcut in enumerate(shortcuts):
            cv2.putText(frame, shortcut, (shortcuts_x + 15, h - 85 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 70)
        print(" " * 15 + "✨ ENHANCED AI AIR DRAWING SYSTEM ✨")
        print("=" * 70)
        print("\n🎨 DRAWING GESTURES:")
        print("  • 1 finger (index) → DRAW")
        print("  • 2 fingers (index + middle) → MOVE CURSOR")
        print("  • Fist (no fingers) → ERASE")
        print("  • 3+ fingers → CHANGE COLOR")
        print("  • All 5 fingers → CLEAR CANVAS")
        print("\n🤟 SIGN LANGUAGE DETECTION:")
        print("  • Hold sign for 2 seconds to add letter")
        print("  • Supported: A, B, C, D, F, I, L, V, W, Y")
        print("\n⌨️  KEYBOARD CONTROLS:")
        print("  • C = Clear canvas")
        print("  • S = Save drawing")
        print("  • + = Increase brush size")
        print("  • - = Decrease brush size")
        print("  • R = Toggle rainbow mode")
        print("  • T = Clear recognized text")
        print("  • Q = Quit")
        print("=" * 70)
        print("\n🎥 Starting camera...\n")
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            print(f"✅ Camera initialized: {w}x{h}")
            print("👋 Ready! Show your hand to start...\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks with custom styling
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2))
                    
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * w)
                    y = int(index_tip.y * h)
                    x, y = self.get_smoothed_position(x, y)
                    
                    self.gesture_mode = self.recognize_gesture(hand_landmarks.landmark)
                    
                    if self.gesture_mode == 'draw':
                        self.draw_line(x, y)
                        cv2.circle(frame, (x, y), self.brush_size + 2, (255, 255, 255), 2)
                        cv2.circle(frame, (x, y), self.brush_size, self.current_color, -1)
                    elif self.gesture_mode == 'select':
                        self.prev_x, self.prev_y = x, y
                        cv2.circle(frame, (x, y), 15, (255, 255, 0), 2)
                        cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
                    elif self.gesture_mode == 'erase':
                        self.erase(x, y)
                        cv2.circle(frame, (x, y), 30, (255, 100, 100), 3)
                    elif self.gesture_mode == 'color_change':
                        self.cycle_color()
                        self.prev_x, self.prev_y = None, None
                    elif self.gesture_mode == 'clear':
                        current_time = time.time()
                        if current_time - self.last_gesture_change > 2.0:
                            self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            self.last_gesture_change = current_time
                            print("🗑️  Canvas cleared by gesture")
                    else:
                        self.prev_x, self.prev_y = None, None
            else:
                self.prev_x, self.prev_y = None, None
                self.gesture_mode = 'idle'
                self.letter_history.clear()
            
            # Combine frame with drawing
            canvas_gray = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.drawing_canvas, self.drawing_canvas, mask=mask)
            combined = cv2.add(frame_bg, canvas_fg)
            
            # Update and draw particle effects
            self.update_particles(combined)
            
            # Draw UI
            self.draw_modern_ui(combined)
            
            cv2.imshow('Enhanced AI Air Drawing System', combined)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Goodbye!")
                break
            elif key == ord('c'):
                self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                print("🗑️  Canvas cleared")
            elif key == ord('s'):
                filename = f'air_drawing_{int(time.time())}.png'
                cv2.imwrite(filename, self.drawing_canvas)
                print(f"💾 Drawing saved as {filename}")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(20, self.brush_size + 1)
                print(f"🖌️  Brush size: {self.brush_size}")
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(1, self.brush_size - 1)
                print(f"🖌️  Brush size: {self.brush_size}")
            elif key == ord('r'):
                self.rainbow_mode = not self.rainbow_mode
                print(f"🌈 Rainbow mode: {'ON' if self.rainbow_mode else 'OFF'}")
            elif key == ord('t'):
                self.recognized_text = ""
                print("🗑️  Recognized text cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n✅ System closed successfully")

if __name__ == "__main__":
    try:
        system = EnhancedAirDrawingSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📦 Make sure you have installed:")
        print("   pip install opencv-python mediapipe numpy")