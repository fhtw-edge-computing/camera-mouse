from tkinter import *  # GUI
from tkinter import messagebox  # Warning Message
import cv2  # Computer Vision
import mediapipe as mp
import time  # can be replaced by datetime
import mouse  # move function of the mouse
from datetime import datetime
import pyautogui  # click function and keyboard simulation


def mouse_move(x, y):
    mouse.move(x, y, absolute=False, duration=0)


def trigger_gesture(action):
    x, y = mouse.get_position()
    print("Gesture triggered: "+action)
    if action == "Left Click":
        pyautogui.click(x, y)
    elif action == "Double Click":
        pyautogui.doubleClick(x, y)
    elif action == "Right Click":
        pyautogui.rightClick(x, y)
    else:
        pyautogui.press("esc")


def save_callback():
    values = [v.get() for v in variables]
    if len(set(values)) != 4:
        messagebox.showwarning("Warning", "All values must be unique")
    else:
        came(values)
        master.destroy()


# Camera Mouse
def came(values):
    cap = cv2.VideoCapture(0)
    pTime = 0  # time when previous frame processed
    mp_draw = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # recognizes only one face at a time
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
    master = Tk()  # create GUI Object
    master.title("Camera Mouse GUI")
    # list with possible actions
    OPTIONS = [
        "Left Click",
        "Double Click",
        "Right Click",
        "Esc button"
    ]
    variables = [StringVar(master) for _ in OPTIONS]
    # list with possible gestures
    ButtonsList = [
        "Smile wide with closed mouth",
        "Open your mouth as saying 'o'",
        "Raising the eyebrows for more than 3 sec",
        "Raising the eyebrows for less than 3 sec"
    ]
    for i, label in enumerate(ButtonsList):
        lab = Label(master, text=label)
        lab.pack()
        variables[i].set(OPTIONS[i])
        w = OptionMenu(master, variables[i], *OPTIONS)  # creates the dropdown menus
        w.pack()  # displays the "w" object in the GUI window
    # add sensitivity
    # add speed
    button = Button(master, text="save", command=save_callback)
    button.pack()
    mainloop()
