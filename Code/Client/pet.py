import cv2
import mediapipe as mp
import numpy as np
from Command import COMMAND as cmd

READ_INTERVAL = 3
RECOG_BUFFER_COUNTS = 1

CAM_CENTER = (200, 160)
REAL_FACE_WIDTH = 15
FOCAL_LENGTH = 410

HEAD_X_MAX = 150
HEAD_X_MIN = 30
HEAD_Y_MAX = 170
HEAD_Y_MIN = 90

ROTATE_THRESHOLD = 5

class Motion:
    SPIN = 1
    FORWARD = 2
    BACKWARD = 3

    def __init__(self):
        pass

    def halt(self):
        end_command = cmd.CMD_MOVE+ "#1#0#0#0#0\n"
        return [end_command]

    def spin(self):
        command = cmd.CMD_MOVE+ "#1#2#0#8#15\n"
        return [command] * 7
    
    def move_forward(self):
        command = cmd.CMD_MOVE+ "#1#0#20#9#0\n"
        return [command]
    
    def move_backward(self):
        command = cmd.CMD_MOVE+ "#1#0#-20#8#0\n"
        return [command]
    
    def gen_action_cmd_queue(self, action):
        assert(action != None)

        if action == self.SPIN:
            return self.spin()
        elif action == self.FORWARD:
            return self.move_forward()
        elif action == self.BACKWARD:
            return self.move_backward()
        
    def move_head_x(self, target_head_x):
        print(f"Moving heax X to {target_head_x:.2f} degrees.")
        command = cmd.CMD_HEAD + f"#1#{int(target_head_x)}\n"
        self.current_head_x = target_head_x
        return [command]

    def move_head_y(self, target_head_y):
        print(f"Moving head Y to {target_head_y:.2f} degrees.")
        command = cmd.CMD_HEAD + f"#0#{int(target_head_y)}\n"
        self.current_head_y = target_head_y
        return [command]

    def rotate_body(self, delta_angle):
        print(f"Rotating body by {delta_angle:.2f} degrees.")
        command = cmd.CMD_MOVE+ f"#1#2#0#8#{max(int(delta_angle), 15)}\n"
        return [command] + self.halt()
        
    def control_robot_head_and_body(self, face_x, face_y, face_depth_cm, bbox_width_pixels):
        commands = []
        error_x_pixels = CAM_CENTER[0] - face_x
        error_y_pixels = CAM_CENTER[1] - face_y
        
        # Step 2: convert pixel error to cm
        pixel_to_cm = REAL_FACE_WIDTH / bbox_width_pixels
        error_x_cm = error_x_pixels * pixel_to_cm
        error_y_cm = error_y_pixels * pixel_to_cm

        # Step 3: calculate desired angle error (in degrees)
        angle_x = np.degrees(np.arctan2(error_x_cm, face_depth_cm))
        angle_y = np.degrees(np.arctan2(error_y_cm, face_depth_cm))

        target_head_x = self.current_head_x + angle_x
        target_head_y = self.current_head_y + angle_y

        print(f"Desired target head angles - X: {target_head_x:.2f}°, Y: {target_head_y:.2f}°")

        if HEAD_X_MIN <= target_head_x <= HEAD_X_MAX:
            if abs(angle_x) >= ROTATE_THRESHOLD:
                commands.extend(self.move_head_x(target_head_x))
        else:
            commands.extend(self.rotate_body(angle_x))

        if HEAD_Y_MIN <= target_head_y <= HEAD_Y_MAX:
            if abs(angle_y) >= ROTATE_THRESHOLD:
                commands.extend(self.move_head_y(target_head_y))
        else:
            print("Y angle too large, ignoring head Y movement.")

        return commands
        
class GestureDetector:
    def __init__(self, client):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        # self.mp_drawing = mp.solutions.drawing_utils
        self.client = client

        self.fram_counter = READ_INTERVAL
        self.recog_buffer_counter = 0
        self.action = None
        self.prev_action = None
        self.halted = True

        self.motion = None

    def process_frame(self, frame):
        self.fram_counter -= 1
        if self.fram_counter >= 0: return
        if self.motion is None:
            self.motion = Motion()
            self.client.cmd_queue = self.motion.move_head_x(90) + self.motion.move_head_y(140)
        self.fram_counter = READ_INTERVAL

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_detection.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                bbox_width = bbox.width * w  # face width in pixels
                bbox_height = bbox.height * h  # face height in pixels
                x_center = int((bbox.xmin + bbox.width / 2) * w)
                y_center = int((bbox.ymin + bbox.height / 2) * h)

                if bbox_width > 0:
                    distance_cm = (REAL_FACE_WIDTH * FOCAL_LENGTH) / bbox_width
                    print(f"Face at ({x_center}, {y_center}), width {bbox_width}, estimated distance: {distance_cm:.2f} cm")
                    if self.client.cmd_queue == []:
                        self.client.cmd_queue = self.motion.control_robot_head_and_body(x_center, y_center, distance_cm, bbox_width)

        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            wrist = landmarks[0]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            def get_finger_direction(tip, base):
                dx = tip.x - base.x
                dy = tip.y - base.y
                return np.arctan2(dy, dx) * 180 / np.pi

            index_angle = get_finger_direction(index_tip, landmarks[6])


            # Gesture 1: Index finger pointing down
            if (
                index_tip.y > wrist.y and  # Index is below wrist
                all(index_tip.y > landmarks[i].y for i in [12, 16, 20]) and
                abs(index_angle - 90) < 45
            ):
                print("Gesture detected: Spin")
                self.action = Motion.SPIN

            # Gesture 2: Open palm, fingers pointing up
            elif (
                all(landmarks[i].y < wrist.y for i in [8, 12, 16, 20]) 
                # and
                # landmarks[8].y < landmarks[6].y and
                # landmarks[12].y < landmarks[10].y
            ):
                print("Gesture detected: Come")
                self.action = Motion.FORWARD

            # Gesture 3: Open palm, fingers pointing down
            elif (
                all(landmarks[i].y > wrist.y for i in [8, 12, 16, 20]) 
                # and
                # landmarks[8].y > landmarks[6].y and
                # landmarks[12].y > landmarks[10].y
            ):
                print("Gesture detected: Go")
                self.action = Motion.BACKWARD

            else:
                print("No Gesture recorded.")
                self.action = None

            # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.action = None

        if self.action != None and self.action == self.prev_action:
            self.recog_buffer_counter += 1
        else:
            self.recog_buffer_counter = 0
            if not self.halted:
                print("Halting")
                self.client.cmd_queue.extend(self.motion.halt())
                self.halted = True
        
        if self.recog_buffer_counter > RECOG_BUFFER_COUNTS:
            self.halted = False
            self.client.cmd_queue = self.motion.gen_action_cmd_queue(self.action)
            
        self.prev_action = self.action