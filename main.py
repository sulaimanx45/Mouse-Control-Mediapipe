import mediapipe.python.solutions.hands as mp
import mediapipe.python.solutions.drawing_utils as utils
import cv2
import time
import pyautogui

hands = mp.Hands(max_num_hands=1)
capture = cv2.VideoCapture(0)
# w=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# h=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out=cv2.VideoWriter('cursor.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30.0,(w,h))
w,h = pyautogui.size()
while True:
    success, frame = capture.read()
    if not success:
        break
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        land_marks = results.multi_hand_landmarks[0]
        land_mark = land_marks.landmark
        pyautogui.moveTo(int(land_mark[8].x*w),int(land_mark[8].y*h))
        if land_mark[12].y<land_mark[11].y:
            pyautogui.click()
            time.sleep(0.2)
        utils.draw_landmarks(frame, land_marks, mp.HAND_CONNECTIONS)
    # out.write(frame)
    cv2.imshow('Hand', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break


capture.release()
# out.release()
cv2.destroyAllWindows()
hands.close()