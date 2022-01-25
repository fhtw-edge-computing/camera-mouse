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
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)  # recognizes only one face at a time
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    mouse_speed = 5
    time_last = datetime.now()
    gesture_delay = 2
    eyebrows_raised_time = None  # if None, it means eyebrows are not raised | if time, eyebrows are raised
    # face detection is running while "Escape" button is being pressed
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                # save the values of the differences from the coordinates
                diff_mouth_width = float(faceLms.landmark[16].y) - float(faceLms.landmark[13].y)
                diff_mouth_length = float(faceLms.landmark[308].x) - float(faceLms.landmark[61].x)
                diff_eyebrows = float(faceLms.landmark[385].y) - float(faceLms.landmark[296].y)
                diff_nose_up = float(faceLms.landmark[152].y) - float(faceLms.landmark[1].y)
                diff_nose_down = float(faceLms.landmark[175].y) - float(faceLms.landmark[19].y)
                diff_nose_left = float(faceLms.landmark[1].x) - float(faceLms.landmark[123].x)
                diff_nose_right = float(faceLms.landmark[352].x) - float(faceLms.landmark[1].x)
                # in case the difference between the two point from the opposite end of the lips is smaller
                # than 0.166325 then the person is not smiling.
                # smile gesture
                if diff_mouth_length > 0.166325 and (datetime.now() - time_last).total_seconds() > gesture_delay:
                    time_last = datetime.now()
                    trigger_gesture(values[0])
                # mouth open gesture
                if diff_mouth_width > 0.038823 and (datetime.now() - time_last).total_seconds() > gesture_delay:
                    time_last = datetime.now()
                    trigger_gesture(values[1])
                if diff_eyebrows > 0.091934:
                    # if eyebrows are being raised for the first time
                    if eyebrows_raised_time is None:
                        # time when first raised
                        eyebrows_raised_time = datetime.now()
                    # eyebrows us for more than 2 sec gesture
                    if (datetime.now() - eyebrows_raised_time).total_seconds() > 2 and (
                            datetime.now() - time_last).total_seconds() > gesture_delay:
                        time_last = datetime.now()
                        trigger_gesture(values[2])
                else:
                    # eyebrows us for less than 2 sec gesture
                    if eyebrows_raised_time is not None and (datetime.now() - eyebrows_raised_time).total_seconds() < 2:
                        trigger_gesture(values[3])
                    eyebrows_raised_time = None
                # mouse move functions
                if diff_nose_right > 0.194273:
                    mouse_move(mouse_speed, 0)
                if diff_nose_left > 0.194273:
                    mouse_move(-1 * mouse_speed, 0)
                if diff_nose_up > 0.28375:
                    mouse_move(0, -1 * mouse_speed)
                if diff_nose_down < 0.210376:
                    mouse_move(0, mouse_speed)
        cTime = time.time()  # current time
        fps = 1 / (cTime - pTime)  # Processing time for this frame = Current time – time when previous frame processed
        pTime = cTime
        # add fps and small text
        cv2.putText(img, "FPS: " + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        cv2.imshow("Camera Mouse", img)
        key = cv2.waitKey(1)
        if key == 27:  # if esc is pressed end the program
            cap.release()
            cv2.destroyAllWindows()
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
    button = Button(master, text="save", command=save_callback)
    button.pack()
    mainloop()
