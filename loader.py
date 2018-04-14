import cv2 as cv
import numpy as np
import tensorflow as tf
# graph = tf.Graph()
# graph.as_default()
sess = tf.Session()
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
saver = tf.train.Saver()
# saver.restore(sess, 'Mymodel.ckpt')
saver.restore(sess,r'./new/model')
print(sess.run(b))
img = cv.imread('myWriting.jpg')
print(img.shape)
a = cv.resize(img,dsize=(28,28))
# a = img
gray = cv.cvtColor(a,cv.COLOR_BGR2GRAY)
print(a.shape)
print(gray.shape)
t,two = cv.threshold(gray,150,255,cv.THRESH_BINARY)
assert isinstance(two,np.ndarray)
for i,v in enumerate(two):
    for j,x in enumerate(v):
        if x == 255:
            two[i][j] = 0
        else:
            two[i][j] = 1
# cv.imwrite('ttt.jpg',a)
# cv.namedWindow('window',cv.WINDOW_AUTOSIZE)
# cv.imshow('window',test)
# cv.waitKey(0)
# print(gray)
# print(type(two))
two = tf.cast(two,tf.float32)

# new = tf.zeros(784)
print(type(two))
new = tf.reshape(two,[1,784])
# t = 0
# for i in two:
#     for j in i:
#         new[t] = j
#         t = t+1
print(sess.run(new))
p = tf.matmul(new,W)+b
print(sess.run(p))
# print(p)
# print(t)