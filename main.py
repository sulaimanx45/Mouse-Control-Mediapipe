import mediapipe.python.solutions.hands as mp
import mediapipe.python.solutions.drawing_utils as utils
import cv2
import time
import pyautogui

hands = mp.Hands(max_num_hands=1)
capture = cv2.VideoCapture(0)
w,h = pyautogui.size()
while True:
    success, frame = capture.read()
    if not success:
        break
    frame=cv2.flip(frame,1)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        land_marks = results.multi_hand_landmarks[0]
        land_mark = land_marks.landmark
        if land_mark[8].y<land_mark[6].y:
            pyautogui.moveTo(int(land_mark[8].x*w),int(land_mark[8].y*h))
        if land_mark[12].y<land_mark[11].y:
            pyautogui.click()
            time.sleep(0.2)
        utils.draw_landmarks(frame, land_marks, mp.HAND_CONNECTIONS)
    cv2.imshow('Hand', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
hands.close()