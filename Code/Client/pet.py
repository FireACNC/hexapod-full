import cv2
import mediapipe as mp
import numpy as np
from Command import COMMAND as cmd

READ_INTERVAL = 10
RECOG_BUFFER_COUNTS = 1

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
        command = cmd.CMD_MOVE+ "#1#0#20#15#0\n"
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
        
class GestureDetector:
    def __init__(self, client):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.client = client

        self.fram_counter = READ_INTERVAL
        self.recog_buffer_counter = 0
        self.action = None
        self.prev_action = None
        self.halted = True

        self.motion = Motion()

    def process_frame(self, frame):
        self.fram_counter -= 1
        if self.fram_counter >= 0: return
        self.fram_counter = READ_INTERVAL

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            y_coords = [lm.y for lm in landmarks]
            x_coords = [lm.x for lm in landmarks]

            min_y = min(y_coords) * h
            max_y = max(y_coords) * h
            min_x = min(x_coords) * w
            max_x = max(x_coords) * w

            body_height = int(max_y - min_y)
            body_position = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))

            print(f"Body detected: Height = {body_height}px, Position = {body_position}")

            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

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
                all(landmarks[i].y < wrist.y for i in [8, 12, 16, 20]) and
                landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y
            ):
                print("Gesture detected: Come")
                self.action = Motion.FORWARD

            # Gesture 3: Open palm, fingers pointing down
            elif (
                all(landmarks[i].y > wrist.y for i in [8, 12, 16, 20]) and
                landmarks[8].y > landmarks[6].y and
                landmarks[12].y > landmarks[10].y
            ):
                print("Gesture detected: Go")
                self.action = Motion.BACKWARD

            else:
                self.action = None

            if self.recog_buffer_counter > RECOG_BUFFER_COUNTS and self.action != None:
                self.halted = False
                self.client.cmd_queue = self.motion.gen_action_cmd_queue(self.action)
            
            # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.action = None

        print("recog buffer:", self.recog_buffer_counter)
        if self.action != None and self.action == self.prev_action:
            self.recog_buffer_counter += 1
        else:
            self.recog_buffer_counter = 0
            if not self.halted:
                print("Halting")
                self.client.cmd_queue.extend(self.motion.halt())
                self.halted = True
        self.prev_action = self.action