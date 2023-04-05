import cv2  # Computer Vision
import time  # can be replaced by datetime
import mouse  # move function of the mouse
import pyautogui  # click function and keyboard simulation
import chime
from queue import Queue
import math
from screeninfo import get_monitors
from enum import Enum

## Eye blink integration
import drowsy_detection

mouse_speed=20
last_nose_pos=(-1,-1)
nose_pos_q=Queue(maxsize=3)

pyautogui.FAILSAFE = False

class MouseMode(Enum):
    REL_MOUSE=1
    REL_JOYSTICK_MOUSE=2
    CURSOR_KEYS=3

mouse_mode=MouseMode.REL_JOYSTICK_MOUSE
frame_counter=0

enabled=True

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
        "EAR_THRESH": 0.415,
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
    if mouse_mode==MouseMode.CURSOR_KEYS:
        if x < -10:
            pyautogui.press("left")
        elif x > 10:
            pyautogui.press("right")
        elif y < -10:
            pyautogui.press("up")
        elif y > 10:
            pyautogui.press("down")
    else:
        mouse.move(x, y, absolute=False, duration=0)

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

def calibrate(vidFrameHandler, frame,state_tracker,nose_pos,head_pose,frame_w,frame_h):
    vidFrameHandler.set_headpose_null(head_pose)
    for m in get_monitors():
        print(str(m))
        if m.is_primary:
            screen_size=(m.width,m.height)
            mouse.move(screen_size[0]/2,screen_size[1]/2,absolute=True,duration=0)

def cam_mouse_EAR():
    global enabled
    global mouse_mode

    cTime=time.time()
    pTime=cTime

    cap = cv2.VideoCapture(0)
    frame_w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fpos_pos=(round(frame_w)-150,30)

    #cap = cv2.VideoCapture("Manschette-test-frontal.mkv")
    vidFrameHandler = drowsy_detection.VideoFrameHandler(cap,state_gesture)

    while True:
        success, img = cap.read()
        #    # Flip the frame horizontally for a selfie-view display.
        img = cv2.flip(img, 1)

        frame, state_tracker, nose_pos, head_pose = vidFrameHandler.process(img,mouse_mode.name)

        if enabled:
            for gesture in state_tracker:
                eye_blinked=gesture["play_alarm"]
                eye_blinked_prev = gesture["play_alarm_prev"]
                if eye_blinked and eye_blinked_prev==False:
                    trigger_gesture(gesture["action"])

                gesture["play_alarm_prev"]=gesture["play_alarm"]

            frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if mouse_mode==MouseMode.REL_MOUSE:
                mouse_move_direct(nose_pos,frame_w,frame_h)
            elif mouse_mode==MouseMode.REL_JOYSTICK_MOUSE or mouse_mode==MouseMode.CURSOR_KEYS:
                mouse_move_joystick_head_pose(head_pose, frame_w, frame_h)

        cTime=time.time()
        fps=round(1/(cTime-pTime))
        pTime=cTime
        cv2.putText(img, f"FPS: {fps}", fpos_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("Camera Mouse", frame)

        key = cv2.pollKey()
        print(key)
        if key == 27:
            break
        elif key == 97:
            enabled=not enabled
        elif key == 99:
            calibrate(vidFrameHandler,frame,state_tracker,nose_pos,head_pose,frame_w,frame_h)
        elif key== 109:
            try:
                mouse_mode=MouseMode(mouse_mode.value+1)
            except:
                mouse_mode=MouseMode(1)

def mouse_move_direct(nose_pos,frame_w,frame_h):
    global last_nose_pos

    if last_nose_pos==(-1,-1):
        last_nose_pos=nose_pos

    avg_nose_pos=((last_nose_pos[0]+nose_pos[0])/2,(last_nose_pos[1]+nose_pos[1])/2)

    diff_nose_pos=(last_nose_pos[0]-avg_nose_pos[0],last_nose_pos[1]-avg_nose_pos[1])
    if(abs(diff_nose_pos[0])>1 or abs(diff_nose_pos[1])> 1):
        mouse_move(-1*diff_nose_pos[0]*mouse_speed,-1*diff_nose_pos[1]*mouse_speed)

    last_nose_pos = avg_nose_pos

def mouse_move_joystick_head_pose(head_pose,frame_w,frame_h):
    global frame_counter

    frame_counter+=1
    if not mouse_mode and not (frame_counter % 12)==0:
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

if __name__ == "__main__":
    cam_mouse_EAR()
