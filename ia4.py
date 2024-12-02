import cv2 as cv

img = cv.imread('imagen.png', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ubb=(24,100,100)
uba=(10,255,255)
uba2= (180, 255,255)

rb = (0,0,255)
ra = (10, 255,255)

mask1 = cv.inRange(hsv, ubb, uba)
mask2 = cv.inRange(hsv, ubb, uba2)
mask = mask1+mask2
res = cv.bitwise_and(img, img, mask=mask)


cv.show('hsv', hsv)
cv.show('img', img)
cv.show('res', res)