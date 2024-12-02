import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    ret, img = cap.read()
    if(ret == True):
        cv.imshow[:2]
        img2 = np.zeros((x,y), dtype='uint8')

        b, g, r = cv.split(img)
        bm = cv.merge([b, img2, img2])
        gm = cv.merge([img2, g, img2])
        rm = cv.merge([img2, img2, r])

        cv.show('b', bm)
        cv.show('g', gm)
        cv.show('r', rm)

        k = cv.waitKey()