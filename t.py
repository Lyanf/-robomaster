import cv2 as cv
import numpy as np

img = cv.imread('gezi.png')
temp = cv.imread('template.png')
window = cv.namedWindow('win')

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img = cv.Canny(img,350,1000)
print(type(img))
t,img = cv.threshold(img,128,255,cv.THRESH_BINARY_INV)
print(type(img))
cv.imshow('win',img)
cv.waitKey(0)