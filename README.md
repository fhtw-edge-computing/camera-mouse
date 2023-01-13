# camera-mouse

## Installation

Execute the following commands to install the program.
Note: On windows activating the virtual environment is done this way: .\venv\Scripts\activate

Tested with Python 3.10.

``` 
  pip -m venv venv
  source ./venv/bin/activate
  pip install -r requirements.txt
```
## Run program

```
   python3 camera-mouse.py
```

## Keyboard shortcuts

### Changing mouse mode

You can change the currently active mode by pressing the key ```m```. Currently there are 3 modes:
* Relative mouse: mouse is moved relatively dependent on head translation (nose tip).
* Relative joystick mouse: mouse is moved dependent on the tilting of the head (nose tip). If the tilt is within a certain dead zone the mouse is not moved. The mouse movement is accelerated dependent on the tilting value and by the following equation:
  * ``x_new = 1.1^yaw``
  * ``y_new=1.1^pitch``
* Cursor key mode: Cursor keys are emulated dependent on the tilting of the head. If the tilt is within a certain dead zone no key is pressed.

### Calibration of head pose

If the head pose values don't represent the real tilting of the head, you can calibrate it by holding the head in a zero tilt position and pressing ``c``.

### Enabling/Disabling Mouse

Press 'a' to enable/disable mouse



