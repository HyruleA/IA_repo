import cv2 as cv

img = cv.read('imagen.png', 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow('img', img)
cv.imshow('gray', gray)
cv.imshow('rgb', rgb)
cv.imshow('hsv', hsv)