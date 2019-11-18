import cv2
import numpy as np
import random

from keras.models import load_model

drawing=False # true if mouse is pressed
mode=True
frame = np.zeros((800,1200,3), np.uint8) + 255
 
# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, frame
 
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
 
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(frame,(current_former_x,current_former_y),(former_x,former_y),(0, 0, 0),3)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(frame,(current_former_x,current_former_y),(former_x,former_y),(0, 0, 0),3)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y

def start():
    global frame
    BORDERS = (0, 255, 0)
    LABELS = (255, 0, 0)
    LIST = (255, 0, 0)

    classes = [
        "banana","house","axe","cup","door","duck","ear","eye","guitar","pizza",
        "alarm_clock","angel","ant","apple","arm","bandage","bat","beach","bed","bicycle",
        "bird","book","cactus","camera","candle","car","carrot","cello","circle","cookie",
        "crab","crayon","dog","donut","face","feather","fish","flamingo","flower","fork",
        "grapes","grass","hand","hat","hexagon","hot_dog","ice_cream","key","knife","octopus"
    ]

    model = load_model('working_model.h5')
    # cap = cv2.VideoCapture(0)

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',paint_draw)
    last_predictions = {}
    while True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [(cv2.boundingRect(c), hierarchy) for c in contours]

        predictions = {}
        for r, h in rectangles:
            if r[2] < 20 and r[3] < 20:
                continue

            # Make the rectangular region around the draw
            leng = int(r[3] * 1.6)
            pt1 = int(r[1] + r[3] // 2 - leng // 2)
            pt2 = int(r[0] + r[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

            # try to resize the image to match sample size
            try:
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            except:
                continue

            final = np.reshape(cv2.dilate(roi, (3, 3, 3)), (1, 1, 28, 28))
            prediction = model.predict_classes(final)[0]
            proba = model.predict_proba(final)[0]
            # if proba[prediction] > 0.50:
            try:
                tmp = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
                predictions[classes[prediction]] = tmp
            except:
                continue

        for label in [x for x in last_predictions if x not in predictions]:
            cv2.destroyWindow(label)

        for label, roi in predictions.items():
            cv2.imshow(label, roi)
        last_predictions = predictions

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key == 27: #Escape KEY
            frame = np.zeros((800,1200,3), np.uint8) + 255

    cv2.destroyAllWindows()

if __name__=="__main__":
    start()
