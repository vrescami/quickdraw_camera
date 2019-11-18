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
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [(cv2.boundingRect(c), hierarchy) for c in contours]

        rois = {i:[] for i in classes}
        for r, h in rectangles:
            if r[2] < 20 and r[3] < 20:
                continue
            cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), BORDERS, 2)

            # Make the rectangular region around the draw
            leng = int(r[3] * 1.6)
            pt1 = int(r[1] + r[3] // 2 - leng // 2)
            pt2 = int(r[0] + r[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

            # try to resize the image to match mnist sample size
            try:
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            except:
                continue

            # final = deskew(cv2.dilate(roi, (1, 1)))
            final = cv2.dilate(roi, (3, 3, 3))
            label = classes[model.predict_classes(np.reshape(final, (1, 1, 28, 28)))[0]]
            rois[label].append(roi)
            cv2.putText(frame, label, (r[0], r[1]),cv2.FONT_HERSHEY_DUPLEX, 1, LABELS, 2)

        # show camera input
        cv2.imshow('frame', frame)

        final_rois = []
        for r in sorted(rois.keys()):
            final_rois.extend(rois[r])

        # show sorted digits
        if final_rois:
            cv2.imshow('roi', np.concatenate(tuple(final_rois), axis=1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    start()
