from tkinter import *  # GUI
from tkinter import messagebox  # Warning Message
import cv2  # Computer Vision
import mediapipe as mp
import time  # can be replaced by datetime
import mouse  # move function of the mouse
from datetime import datetime
import pyautogui  # click function and keyboard simulation
import chime
from queue import Queue
import math

## Eye blink integration

import drowsy_detection

mouse_speed=10
last_nose_pos=(-1,-1)
nose_pos_q=Queue(maxsize=3)

pyautogui.FAILSAFE = False
mouse_mode=False
frame_counter=0

# Define gestures by selecting the landmark indexes which shall be used for calculation the EAR value: https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/
# The landmark indexes can be found here (left / right must be flipped because of selfie view): https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

state_gesture=[{
        "label": "Mouth Open",
        "gesture_idxs": [57, 37, 267, 287, 314, 84],
        "EAR_THRESH": 0.40,
        "WAIT_TIME": 0.6,
        "operator": ">",
        "start_time": time.perf_counter(),
        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
        "COLOR": drowsy_detection.GREEN,
        "play_alarm": False,
        "play_alarm_prev": False,
        "action": "Double Click"
    },
    {
        "label": "Eye_r",
        "gesture_idxs": [362, 385, 387, 263, 373, 380],
        "EAR_THRESH": 0.18,
        "WAIT_TIME": 0.6,
        "operator": "<",
        "start_time": time.perf_counter(),
        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
        "COLOR": drowsy_detection.GREEN,
        "play_alarm": False,
        "play_alarm_prev": False,
        "action": "Right Click"
    },
    {
        "label": "Eye_l",
        "gesture_idxs": [33, 160, 158, 133, 153, 144],
        "EAR_THRESH": 0.18,
        "WAIT_TIME": 0.6,
        "operator": "<",
        "start_time": time.perf_counter(),
        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
        "COLOR": drowsy_detection.GREEN,
        "play_alarm": False,
        "play_alarm_prev": False,
        "action": "Left Click"
    },
#    {
#        "label": "brow_up_r",
#        "gesture_idxs": [33, 334, 296, 133, 153, 144],
#        "EAR_THRESH": 2.9,
#        "WAIT_TIME": 0.6,
#        "operator": ">",
#        "start_time": time.perf_counter(),
#        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
#        "COLOR": drowsy_detection.GREEN,
#        "play_alarm": False,
#        "play_alarm_prev": False,
#        "action": ""
#    },
    {
        "label": "brow_up_l",
#        "gesture_idxs": [33, 105, 66, 133, 153, 144],
        "gesture_idxs": [33, 105, 66, 263, 153,144],
        "EAR_THRESH": 0.43,
        "WAIT_TIME": 0.6,
        "operator": ">",
        "start_time": time.perf_counter(),
        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
        "COLOR": drowsy_detection.GREEN,
        "play_alarm": False,
        "play_alarm_prev": False,
        "action": "KEY_SPACE"
    },
    # {
    #     "label": "cheek_up",
    #     "gesture_idxs": [103, 67, 109, 10, 36, 205],
    #     "EAR_THRESH": 2.5,
    #     "WAIT_TIME": 0.6,
    #     "operator": "<",
    #     "start_time": time.perf_counter(),
    #     "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
    #     "COLOR": drowsy_detection.GREEN,
    #     "play_alarm": False,
    #     "play_alarm_prev": False,
    #     "action": ""
    # }
]

def mouse_move(x, y):
    #print("mouse move: x={0}, y={1}".format(x,y))
    if mouse_mode:
        mouse.move(x, y, absolute=False, duration=0)
    else:
        if x < 0:
            pyautogui.press("left")
        if x > 0:
            pyautogui.press("right")
        if y < 0:
            pyautogui.press("up")
        if y > 0:
            pyautogui.press("down")

def trigger_gesture(action):
    x, y = mouse.get_position()
    print("Gesture triggered: "+action)
    if action == "Left Click":
        pyautogui.click(x, y)
        chime.success()
    elif action == "Double Click":
        pyautogui.doubleClick(x, y)
        chime.success()
        chime.success()
    elif action == "Right Click":
        pyautogui.rightClick(x, y)
        chime.warning()
    elif action == "Middle Click":
        pyautogui.middleClick(x,y)
        chime.info()
    elif action == "KEY_SPACE":
        pyautogui.press("space")
        chime.error()

def save_callback():
    values = [v.get() for v in variables]
    if len(set(values)) != 4:
        messagebox.showwarning("Warning", "All values must be unique")
    else:
        # came(values)
        cam_mouse_EAR()
        master.destroy()

def cam_mouse_EAR():
    cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("taster-rotate+cut-lachen2.mp4")
    vidFrameHandler = drowsy_detection.VideoFrameHandler(cap,state_gesture)

    while True:
        success, img = cap.read()
        #    # Flip the frame horizontally for a selfie-view display.
        img = cv2.flip(img, 1)
        frame, state_tracker, nose_pos, head_pose = vidFrameHandler.process(img)

        for gesture in state_tracker:
            eye_blinked=gesture["play_alarm"]
            eye_blinked_prev = gesture["play_alarm_prev"]
            if eye_blinked and eye_blinked_prev==False:
                trigger_gesture(gesture["action"])

            gesture["play_alarm_prev"]=gesture["play_alarm"]

        handle_mouse_action(head_pose,cap)
        cv2.imshow("Camera Mouse", frame)

        key = cv2.pollKey()
        if key == 27:
            break

def handle_mouse_action(nose_pos, cap):
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #mouse_move_joystick(nose_pos,frame_w,frame_h)
    mouse_move_joystick_head_pose(nose_pos,frame_w,frame_h)

def mouse_move_direct(nose_pos,frame_w,frame_h):
    global last_nose_pos

    if last_nose_pos==(-1,-1):
        last_nose_pos=nose_pos

    avg_nose_pos=((last_nose_pos[0]+nose_pos[0])/2,(last_nose_pos[1]+nose_pos[1])/2)

    diff_nose_pos=(last_nose_pos[0]-avg_nose_pos[0],last_nose_pos[1]-avg_nose_pos[1])
    if(abs(diff_nose_pos[0])>2 or abs(diff_nose_pos[1])> 2):
        mouse_move(diff_nose_pos[0]*mouse_speed,-1*diff_nose_pos[1]*mouse_speed)

    last_nose_pos = avg_nose_pos

def mouse_move_joystick_head_pose(head_pose,frame_w,frame_h):
    global frame_counter

    frame_counter+=1
    if not mouse_mode and not (frame_counter % 8)==0:
        return

    pitch=head_pose[0]
    yaw=head_pose[1]

    mouse_speed_co=1.1
    mouse_speed_max=25
    acceleration=3

    threshold=(-2,2,-0,2)

    if not mouse_mode:
        threshold=(-6,6,-4,6)

    mouse_speed_x=0
    mouse_speed_y=0

    # See where the user's head tilting
    if yaw < threshold[0]:
        text = "Looking Left"
        mouse_speed_x = -1*min(math.pow(mouse_speed_co, abs(yaw*acceleration)),mouse_speed_max)
    if yaw > threshold[1]:
        text = "Looking Right"
        mouse_speed_x = min(math.pow(mouse_speed_co, abs(yaw*acceleration)), mouse_speed_max)
    if pitch < threshold[2]:
        text = "Looking Down"
        mouse_speed_y = min(math.pow(mouse_speed_co, abs(pitch*acceleration)), mouse_speed_max)
    if pitch > threshold[3]:
        text = "Looking Up"
        mouse_speed_y = -1*min(math.pow(mouse_speed_co, abs(pitch*acceleration)), mouse_speed_max)

    #print(text)
    mouse_move(mouse_speed_x, mouse_speed_y)

def mouse_move_joystick(nose_pos,frame_w,frame_h):
    global last_nose_pos

    mouse_speed_co=1.1
    thresh_percentage=(0.55,0.45)
    mouse_speed_max=20
    thresh_pixel=(frame_w*thresh_percentage[0],frame_w*thresh_percentage[1],frame_h*thresh_percentage[0],frame_h*thresh_percentage[1])

    mouse_speed_x=0
    mouse_speed_y=0

    if(nose_pos[0]<thresh_pixel[0]):
        mouse_speed_x = -1*min(math.pow(mouse_speed_co, abs(nose_pos[0]-thresh_pixel[0])),mouse_speed_max)
        #mouse_move(-1 * mouse_speed[0], 0)
    if(nose_pos[0]>thresh_pixel[1]):
        mouse_speed_x = min(math.pow(mouse_speed_co, abs(nose_pos[0] - thresh_pixel[1])),mouse_speed_max)
        #mouse_move(1 * mouse_speed[0], 0)
    if (nose_pos[1] > thresh_pixel[2]):
        mouse_speed_y = min(math.pow(mouse_speed_co, abs(nose_pos[1] - thresh_pixel[2])),mouse_speed_max)
        #mouse_move(0,1 * mouse_speed[1])
    if(nose_pos[1]<thresh_pixel[3]):
        mouse_speed_y = -1*min(math.pow(mouse_speed_co, abs(nose_pos[1] - thresh_pixel[3])),mouse_speed_max)
        #mouse_move(0,-1 * mouse_speed[1])

    mouse_move(mouse_speed_x,mouse_speed_y)

# Camera Mouse
def came(values):
    cap = cv2.VideoCapture(2)
    pTime = 0  # time when previous frame processed
    mp_draw = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    # recognizes only one face at a time
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)
    mouse_speed = 5
    time_last = datetime.now()
    gesture_delay = 2
    eyebrows_raised_time = None  # if None, it means eyebrows are not raised | if time, eyebrows are raised
    # face detection is running while "Escape" button is being pressed
    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mp_draw.draw_landmarks(img, faceLms, mp_face_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)
                # save the values of the differences from the coordinates
                diff_mouth_width_y = float(faceLms.landmark[16].y) - float(faceLms.landmark[13].y)
                diff_mouth_length_x = float(faceLms.landmark[308].x) - float(faceLms.landmark[61].x)
                diff_eyebrows_y = float(faceLms.landmark[385].y) - float(faceLms.landmark[296].y)
                diff_nose_up_y = float(faceLms.landmark[152].y) - float(faceLms.landmark[1].y)
                diff_nose_down_y = float(faceLms.landmark[175].y) - float(faceLms.landmark[19].y)
                diff_nose_left_x = float(faceLms.landmark[1].x) - float(faceLms.landmark[123].x)
                diff_nose_right_x = float(faceLms.landmark[352].x) - float(faceLms.landmark[1].x)
                face_length_y = float(faceLms.landmark[152].y) - float(faceLms.landmark[10].y)
                face_width_x = float(faceLms.landmark[352].x) - float(faceLms.landmark[123].x)

                # Normalization: Divide horizontal mouth width by face_width_x to normalize the value (make it invariant to head size)
                print("diff_mout_length_x: "+str(diff_mouth_length_x / face_width_x)+", default mouth x: "+str(0.1 / face_width_x))

                if (diff_mouth_length_x / face_width_x ) > 0.42:
                    # and (datetime.now() - time_last).total_seconds() > gesture_delay:
                    print("Smile triggered")
                    time_last = datetime.now()
                    trigger_gesture(values[0])
                if diff_mouth_width_y > (face_width_x * 0.121227) and (datetime.now() - time_last).total_seconds() > gesture_delay:
                    time_last = datetime.now()
                    trigger_gesture(values[1])
                if diff_eyebrows_y > (face_width_x * 0.287071):
                    # if eyebrows are being raised for the first time
                    if eyebrows_raised_time is None:
                        # time when first raised
                        #
                        eyebrows_raised_time = datetime.now()
                    # eyebrows up for more than 2 sec gesture

                    if (datetime.now() - eyebrows_raised_time).total_seconds() > 2 and (datetime.now() - time_last).total_seconds() > gesture_delay:
                        time_last = datetime.now()
                        trigger_gesture(values[2]) # this gesture is not always triggered
                else:
                    # eyebrows up for less than 2 sec gesture
                    if eyebrows_raised_time is not None and (datetime.now() - eyebrows_raised_time).total_seconds() < 2:
                        time_last = datetime.now()
                        trigger_gesture(values[3])
                    eyebrows_raised_time = None

                if diff_nose_right_x > (face_width_x * 0.626632):
                    time_last = datetime.now()
                    mouse_move(mouse_speed, 0)
                # if diff_nose_left_x > 0,194273:
                if diff_nose_left_x > (face_width_x * 0.586632):
                    time_last = datetime.now()
                    mouse_move(-1 * mouse_speed, 0)
                # if diff_nose_up_y > 0,28375: 0,397314
                if diff_nose_up_y > (face_width_x * 0.746032):
                    time_last = datetime.now()
                    mouse_move(0, -1 * mouse_speed)
                # if diff_nose_down_y < 0,210376: 0,429268
                if diff_nose_down_y < (face_width_x * 0.806032):
                    time_last = datetime.now()
                    mouse_move(0, mouse_speed)

        # -1 mirrors the image in order to provide user friendly interface
        cv2.imshow("Camera Mouse", img[:, ::-1, :])

        key = cv2.pollKey()
        if key == 27:
            break



if __name__ == "__main__":
    #master = Tk()  # create GUI Object
    #master.title("Camera Mouse GUI")
    # list with possible actions
    #OPTIONS = [
    #    "Left Click",
    #    "Double Click",
    #    "Right Click",
    #    "Esc button"
    #]
    #variables = [StringVar(master) for _ in OPTIONS]
    # list with possible gestures
    #ButtonsList = [
    #    "Smile wide with closed mouth",
    #    "Open your mouth as saying 'o'",
    #    "Raising the eyebrows for more than 3 sec",
    #    "Raising the eyebrows for less than 3 sec"
    #]
    #for i, label in enumerate(ButtonsList):
    #    lab = Label(master, text=label)
    #    lab.pack()
    #    variables[i].set(OPTIONS[i])
    #    w = OptionMenu(master, variables[i], *OPTIONS)  # creates the dropdown menus
    #    w.pack()  # displays the "w" object in the GUI window
    # add sensitivity
    # add speed
    #button = Button(master, text="save", command=save_callback)
    #button.pack()
    #mainloop()
    cam_mouse_EAR()
