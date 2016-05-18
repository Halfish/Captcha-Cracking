#!/usr/bin/env python
# coding=utf-8

import cv2
import os.path

def preprocess_nacao_type3(dirname, index):
    filename = os.path.join(dirname, index + '.jpg')
    img = cv2.imread(filename)
    r, g, b = cv2.split(img)
    blur = cv2.bilateralFilter(b, 7, 35, 35)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(dirname, 'b' + index + '.jpg'), adaptive)

def nacao_type3():
    dirname = '../trainpic/nacao/type3/'
    for i in range(1000):
        preprocess_nacao_type3(dirname, str(i+1))
    pass

if __name__ == '__main__':
    nacao_type3()
