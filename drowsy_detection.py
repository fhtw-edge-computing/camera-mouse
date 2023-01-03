import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

# Used for coloring landmark points.
# Its value depends on the current EAR value.
RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)  # BGR

def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_nose_pos(landmarks, refer_idx, frame_width, frame_height):
    lm = landmarks[refer_idx]
    coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
    return coord

def calculate_avg_ear(landmarks, gesture_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    ear, lm_coordinates = get_ear(landmarks, gesture_idxs, image_w, image_h)

    return ear, (lm_coordinates)


def plot_eye_landmarks(frame, lm_coordinates, color):
    if lm_coordinates:
        for coord in lm_coordinates:
            cv2.circle(frame, coord, 2, color, -1)

    #frame = cv2.flip(frame, 1)
    return frame

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


def get_head_pose(frame, landmarks, img_w, img_h):
    face_3d = []
    face_2d = []

    for idx in [33,263,1,61,291,199]:
        lm=landmarks[idx]
        if idx == 1:
            nose_2d = (lm.x * img_w, lm.y * img_h)
            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)

        x, y = int(lm.x * img_w), int(lm.y * img_h)

        # Get the 2D Coordinates
        face_2d.append([x, y])

        # Get the 3D Coordinates
        face_3d.append([x, y, lm.z])

    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Display the nose direction
    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 5), int(nose_2d[1] - x * 5))

    cv2.line(frame, p1, p2, (255, 0, 0), 3)
    head_pose=(x,y,z)
    return head_pose

class VideoFrameHandler:
    def __init__(self, cap, state_gesture):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        self.frame_w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Left and right eye chosen landmarks.
        self.lm_idxs = [
            {
            "left": [57, 37, 267, 287, 314, 84],
            "right": [57, 37, 267, 287, 314, 84],
        }
        #  {
         #  "left": [362, 385, 387, 263, 373, 380],
          # "right": [33, 160, 158, 133, 153, 144],
        #}
        ]


        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = state_gesture

        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        if hasattr(frame, 'flags'):
            frame.flags.writeable = False

        #frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(self.frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(self.frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)
        nose_pos=(self.frame_w/2,self.frame_h/2)
        head_pose=(0.0,0.0,0.0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            head_pose=get_head_pose(frame,landmarks,self.frame_w,self.frame_h)
            for state_tracker, idx in zip(self.state_tracker, range(len(self.state_tracker))):
                EAR, lm_coordinates = calculate_avg_ear(landmarks, state_tracker["gesture_idxs"], self.frame_w, self.frame_h)
                frame = plot_eye_landmarks(frame, lm_coordinates, state_tracker["COLOR"])
                nose_pos=get_nose_pos(landmarks,1,self.frame_w,self.frame_h)
                cv2.circle(frame, nose_pos, 2,mp.solutions.drawing_utils.BLUE_COLOR, -1)

                local_map={
                    "EAR":EAR,
                    "thresh":state_tracker["EAR_THRESH"]
                }
                if eval("EAR" +state_tracker["operator"]+"thresh",local_map):

                    # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                    # and reset the start_time for the next iteration.
                    end_time = time.perf_counter()

                    state_tracker["DROWSY_TIME"] += end_time - state_tracker["start_time"]
                    state_tracker["start_time"] = end_time
                    state_tracker["COLOR"] = RED

                    if state_tracker["DROWSY_TIME"] >= state_tracker["WAIT_TIME"]:
                        state_tracker["play_alarm_prey"] = state_tracker["play_alarm"]
                        state_tracker["play_alarm"] = True
                        plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, state_tracker["COLOR"])

                else:
                    self.reset_state(state_tracker)

                EAR_txt = f"{state_tracker['label']}: {round(EAR, 2)}, time: {round(state_tracker['DROWSY_TIME'], 3)} Secs"
                #DROWSY_TIME_txt = f"DROWSY: {round(state_tracker['DROWSY_TIME'], 3)} Secs"
                txt_pos=self.EAR_txt_pos
                txt_pos=(txt_pos[0],txt_pos[1]+idx*30)
                plot_text(frame, EAR_txt, txt_pos, state_tracker["COLOR"])
                #plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos+idx*(0,10), state_tracker["COLOR"])

        else:
            for state_tracker in self.state_tracker:
                self.reset_state(state_tracker)

        # Flip the frame horizontally for a selfie-view display.
        #frame = cv2.flip(frame, 1)

        return frame, self.state_tracker, nose_pos, head_pose

    def reset_state(self, state_tracker):
        state_tracker["start_time"] = time.perf_counter()
        state_tracker["DROWSY_TIME"] = 0.0
        state_tracker["COLOR"] = GREEN
        state_tracker["play_alarm_prev"] = state_tracker["play_alarm"]
        state_tracker["play_alarm"] = False


