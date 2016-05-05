#!/usr/bin/env python
# coding=utf-8

import cv2
import os
import numpy as np
import argparse

'''
Usage 1: to load picutures from ./trainpic/ and cut them into 3 pieces
Usage 2: util tool for type4_predict.lua to preprocess pictures
    for example:
    > 1. python cut.py nx      # this will dump ningxia data
    > 2. python cut.py nx -f single -p ../trainpic/type4/ningxia/1.jpg
'''

def norByMoments(img):
    norImage = 255 - img    # must be white words with black background
    moments = cv2.moments(norImage)
    (centroid_x, centroid_y) = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    (y1, y2, x1, x2) = (0, img.shape[0], 0, img.shape[1])
    if (centroid_x * 2 > x2):
        x1 = centroid_x * 2 - x2
    else:
        x2 = centroid_x * 2
    if (centroid_y * 2 > y2):
        y1 = centroid_y * 2 - y2
    else:
        y2 = centroid_y * 2
    norImage = 255 - norImage[y1:y2, x1:x2]
    return norImage

def uniformSize(img):
    '''
    normalize the image into size of 30 * 30 pixes
    '''
    uni_size_x = 30
    uni_size_y = 30
    uniImage = 255 * np.ones((uni_size_y, uni_size_x), np.uint8)
    if len(img.shape) == 3:
        uniImage = 255 * np.ones((uni_size_y, uni_size_x, 3), np.uint8)
    y1 = 15 - img.shape[0] / 2
    y2 = y1 + img.shape[0]
    x1 = 15 - img.shape[1] / 2
    x2 = x1 + img.shape[1]
    uniImage[y1:y2, x1:x2] = img
    return uniImage

def preprocess_chq(n_img, n_alpha, n_beta, n_gamma):
    '''
    cut the image into three individual pieces for recognition
    '''
    img = cv2.imread(n_img)
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    word_pix = np.argsort(-hist_r.astype(int), axis=0)[1][0]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if not(img[y][x][0] == word_pix and img[y][x][1] == word_pix and img[y][x][2] == word_pix):
                img[y][x][0] = 255
                img[y][x][1] = 255
                img[y][x][2] = 255
    alpha = img[10:40, 0:16]
    beta = img[10:40, 20:45]
    gamma = img[10:40, 44:57]
    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    beta = cv2.cvtColor(beta, cv2.COLOR_BGR2GRAY)
    gamma = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
    alpha = uniformSize(norByMoments(alpha))
    beta = uniformSize(norByMoments(beta))
    gamma = uniformSize(norByMoments(gamma))
    cv2.imwrite(n_alpha, alpha)
    cv2.imwrite(n_beta, beta)
    cv2.imwrite(n_gamma, gamma)

def chongqing():
    num = 99
    label_path = '../trainpic/type4/chongqing/label.txt'
    pic_dir = '../trainpic/type4/chongqing/'
    save_dir = '../trainpic/type4/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.png')
            n_alpha = os.path.join(save_dir, 'chq_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta = os.path.join(save_dir, 'chq_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'chq_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_chq(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

def preprocess_gs(n_img, n_alpha, n_beta, n_gamma):
    img = cv2.imread(n_img)
    alpha = img[25:43, 2:20]
    beta = img[25:43, 18:36]
    gamma = img[25:43, 36:54]
    cv2.imwrite(n_alpha, alpha)
    cv2.imwrite(n_beta, beta)
    cv2.imwrite(n_gamma, gamma)

def gansu():
    num = 99
    label_path = '../trainpic/type4/gansu/label.txt'
    pic_dir = '../trainpic/type4/gansu/'
    save_dir = '../trainpic/type4/'
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.jpg')
            n_alpha = os.path.join(save_dir, 'gs_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta= os.path.join(save_dir, 'gs_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'gs_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_gs(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

def preprocess_nx(n_img, n_alpha, n_beta, n_gamma):
    img = cv2.imread(n_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bBlur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(bBlur, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[8:27, 20:36]
    beta = thresh[12:27, 44:62]
    gamma = thresh[8:27, 70:85]
    cv2.imwrite(n_alpha, uniformSize(alpha))
    cv2.imwrite(n_beta, uniformSize(beta))
    cv2.imwrite(n_gamma, uniformSize(gamma))

def ningxia():
    num = 99
    label_path = '../trainpic/type4/ningxia/label.txt'
    pic_dir = '../trainpic/type4/ningxia/'
    save_dir = '../trainpic/type4/'
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.jpg')
            n_alpha = os.path.join(save_dir, 'nx_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta = os.path.join(save_dir, 'nx_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'nx_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_nx(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

def preprocess_tj(n_img, n_alpha, n_beta, n_gamma):
    img = cv2.imread(n_img)
    b, g, r = cv2.split(img)
    bBlur = cv2.bilateralFilter(g, 9, 75, 75)
    ret, thresh = cv2.threshold(bBlur, 160, 255, cv2.THRESH_BINARY)
    alpha = thresh[2:25, 10:25]
    beta = thresh[2:25, 42:62]
    gamma = thresh[2:25, 70:85]
    cv2.imwrite(n_alpha, uniformSize(alpha))
    cv2.imwrite(n_beta, uniformSize(beta))
    cv2.imwrite(n_gamma, uniformSize(gamma))

def tianjin():
    num = 99
    label_path = '../trainpic/type4/tianjin/label.txt'
    pic_dir = '../trainpic/type4/tianjin/'
    save_dir = '../trainpic/type4/'
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.jpg')
            n_alpha = os.path.join(save_dir, 'tj_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta= os.path.join(save_dir, 'tj_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'tj_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_tj(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

def preprocess_jx(n_img, n_alpha, n_beta, n_gamma):
    img = cv2.imread(n_img)
    bBlur = cv2.bilateralFilter(img, 9, 175, 175)
    ret, thresh = cv2.threshold(bBlur, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[3:33, 13:38]
    beta = thresh[0:35, 35:70]
    gamma = thresh[3:33, 65:90]
    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    beta = cv2.cvtColor(beta, cv2.COLOR_BGR2GRAY)
    gamma = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(n_alpha, alpha)
    cv2.imwrite(n_beta, beta)
    cv2.imwrite(n_gamma, gamma)

def jiangxi():
    num = 99
    label_path = '../trainpic/type4/jiangxi/label.txt'
    pic_dir = '../trainpic/type4/jiangxi/'
    save_dir = '../trainpic/type4/'
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.jpg')
            n_alpha = os.path.join(save_dir, 'jx_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta= os.path.join(save_dir, 'jx_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'jx_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_jx(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

def preprocess_small(n_img, n_alpha, n_beta, n_gamma):
    # small captcha, 80*30, shanxi, sichuan, xinjiang
    img = cv2.imread(n_img, 0)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[3:18, 2:13]      # 15 * 11
    beta = thresh[3:19, 18:34]      # 16 * 16
    gamma = thresh[3:18, 32:43]     # 15 * 11
    cv2.imwrite(n_alpha, alpha)
    cv2.imwrite(n_beta, beta)
    cv2.imwrite(n_gamma, gamma)

def small():
    # small captcha, 80*30, shanxi, sichuan, xinjiang
    num = 200
    label_path = '../trainpic/type4/small/label.txt'
    pic_dir = '../trainpic/type4/small/'
    save_dir = '../trainpic/type4/'
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for i in range(num):
            n_img = os.path.join(pic_dir, str(i+1) + '.jpg')
            n_alpha = os.path.join(save_dir, 'small_'+ str(i+1) + '_1_' + labels[i][0] + '.png')
            n_beta= os.path.join(save_dir, 'small_'+ str(i+1) + '_2_' + labels[i][1] + '.png')
            n_gamma = os.path.join(save_dir, 'small_'+ str(i+1) + '_3_' + labels[i][2] + '.png')
            preprocess_small(n_img, n_alpha, n_beta, n_gamma)
        pass
    pass

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('province', choices=['chq', 'gs', 'nx', 'tj', 'jx', 'small'], help='which province to choice')
    parser.add_argument('-f', '--function', choices=['single', 'dump'], default='dump',
                        help='cut single picture or dump data')
    parser.add_argument('-p', '--imgpath', help='image path to read from')
    args = parser.parse_args()
    if args.function == 'dump':
        mapper = {'chq':chongqing, 'gs':gansu, 'nx':ningxia, 'tj':tianjin, 'jx':jiangxi, 'small':small}
        f = mapper[args.province]
        f()
    elif args.function == 'single':
        mapper = {'chq':preprocess_chq, 'gs':preprocess_gs, 'nx':preprocess_nx, 'tj':preprocess_tj,
                  'jx':preprocess_jx, 'small':preprocess_small}
        f = mapper[args.province]
        f(args.imgpath, 'alpha.png', 'beta.png', 'gamma.png')
    pass
