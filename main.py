import cv2
import numpy as np
import random

from keras.models import load_model

def start():
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
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [(cv2.boundingRect(c), hierarchy) for c in contours]

        guesses = ['banana', 'house', 'candle', 'fork', 'pizza']

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
            if proba[prediction] > 0.70:
                label = classes[prediction]
                cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), BORDERS, 2)
                cv2.putText(frame, label, (r[0], r[1]),cv2.FONT_HERSHEY_DUPLEX, 1, LABELS, 2)
                if label in guesses:
                    cv2.imshow(label, roi)

        # show camera input
        cv2.imshow('frame', frame)
        cv2.imshow("threshold", im_th)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    start()
