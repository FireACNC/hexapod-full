import cv2
import mediapipe as mp
import numpy as np
from Command import COMMAND as cmd

READ_INTERVAL = 10

class GestureDetector:
    def __init__(self, client):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.client = client

        self.fram_counter = READ_INTERVAL
        self.spin_counter = 0

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
            for hand_landmarks in hand_results.multi_hand_landmarks:
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
                    print("Gesture detected: Spin (Index pointing down)")
                    self.spin_counter += 1

                    if self.spin_counter >= 2:
                        command = cmd.CMD_MOVE+ "#1#2#0#8#15\n"
                        end_command = cmd.CMD_MOVE+ "#1#0#0#0#0\n"
                        self.client.cmd_queue = [command] * 7 + [end_command]

                else: 
                    self.spin_counter = 0
                    # Gesture 2: Open palm, fingers pointing up
                    if (
                        all(landmarks[i].y < wrist.y for i in [8, 12, 16, 20]) and
                        landmarks[8].y < landmarks[6].y and
                        landmarks[12].y < landmarks[10].y
                    ):
                        print("Gesture detected: Come (Palm open, fingers up)")

                    # Gesture 3: Open palm, fingers pointing down
                    elif (
                        all(landmarks[i].y > wrist.y for i in [8, 12, 16, 20]) and
                        landmarks[8].y > landmarks[6].y and
                        landmarks[12].y > landmarks[10].y
                    ):
                        print("Gesture detected: Go (Palm open, fingers down)")

                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)



# def detect_body_and_hand_gestures():
#     mp_pose = mp.solutions.pose
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils

#     pose = mp_pose.Pose()
#     hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         pose_results = pose.process(frame_rgb)
#         hand_results = hands.process(frame_rgb)

#         # Pose detection
#         if pose_results.pose_landmarks:
#             landmarks = pose_results.pose_landmarks.landmark
#             y_coords = [lm.y for lm in landmarks]
#             x_coords = [lm.x for lm in landmarks]

#             min_y = min(y_coords) * h
#             max_y = max(y_coords) * h
#             min_x = min(x_coords) * w
#             max_x = max(x_coords) * w

#             body_height = int(max_y - min_y)
#             body_position = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))

#             print(f"Body detected: Height = {body_height}px, Position = {body_position}")

#             mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Hand gesture detection
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 landmarks = hand_landmarks.landmark

#                 wrist = landmarks[0]
#                 index_tip = landmarks[8]
#                 middle_tip = landmarks[12]
#                 ring_tip = landmarks[16]
#                 pinky_tip = landmarks[20]
#                 palm_center_y = np.mean([landmarks[i].y for i in [0, 5, 9, 13, 17]])

#                 def get_finger_direction(tip, base):
#                     dx = tip.x - base.x
#                     dy = tip.y - base.y
#                     return np.arctan2(dy, dx) * 180 / np.pi

#                 index_angle = get_finger_direction(index_tip, landmarks[6])

#                 # Gesture 1: Index finger pointing down
#                 if (
#                     index_tip.y > wrist.y and  # Index is below wrist
#                     all(index_tip.y > landmarks[i].y for i in [12, 16, 20]) and
#                     abs(index_angle - 90) < 45
#                 ):
#                     print("Gesture detected: Spin (Index pointing down)")

#                 # Gesture 2: Open palm, fingers pointing up
#                 elif (
#                     all(landmarks[i].y < wrist.y for i in [8, 12, 16, 20]) and
#                     landmarks[8].y < landmarks[6].y and
#                     landmarks[12].y < landmarks[10].y
#                 ):
#                     print("Gesture detected: Come (Palm open, fingers up)")

#                 # Gesture 3: Open palm, fingers pointing down
#                 elif (
#                     all(landmarks[i].y > wrist.y for i in [8, 12, 16, 20]) and
#                     landmarks[8].y > landmarks[6].y and
#                     landmarks[12].y > landmarks[10].y
#                 ):
#                     print("Gesture detected: Go (Palm open, fingers down)")

#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         cv2.imshow("Body and Gesture Detection", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Run the function
# if __name__ == "__main__":
#     detect_body_and_hand_gestures()
