#!/usr/bin/env python
# coding=utf-8
import cv2
import PIL
import base64
import numpy as np
import cStringIO
import pytesseract
import Image

from functools import wraps
from time import time
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print 'func:%r took: %2.4f ms' %(f.__name__, (te - ts) * 1000)
        return result
    return wrap


type4_province = ['chongqing', 'gansu', 'jiangxi', 'ningxia', 'tianjin',
                  'shan3xi', 'sichuan', 'xinjiang', 'beijing']
type4_mapper = {'chongqing':'chq', 'gansu':'gs', 'jiangxi':'jx', 'ningxia':'nx', 'tianjin':'tj',
                'shan3xi':'small', 'sichuan':'small', 'xinjiang':'small', 'beijing':'single'}
svhn_province = ['anhui', 'guangxi', 'henan', 'heilongjiang', 'qinghai',
                 'shanxi', 'xizang', 'fujian', 'nation', 'hebei', 'shanghai',
                 'yunnan', 'hunan', 'guangdong', 'hainan', 'neimenggu', 'nacao',
                 'guizhou', 'jilin', 'shandong']
tess_province = ['jiangsu', 'liaoning']

def crack(imgfile, province):
    '''
    given a binary image file and province name
    return json string
    '''
    pil_image = PIL.Image.open(cStringIO.StringIO(imgfile))

    if province in type4_province:
        return 'type4', pil_image.format, type4_mapper[province], type4_cut(pil_image, province)
    elif province in svhn_province:
       return 'svhn', pil_image.format, province, [svhn(pil_image,province)]
    elif province in tess_province:
        return 'tess', pil_image.format, province, tess_reco(pil_image, province)


def type4_cut(pil_image, province):
    # cut the image and encode to base64 strings
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    mapper = {'chongqing':chq, 'gansu':gs, 'jiangxi':jx, 'ningxia':nx, 'tianjin':tj,
              'shan3xi':small, 'sichuan':small, 'xinjiang':small, 'beijing':beijing}
    patches = mapper[province](img)
    for i in range(len(patches)):
        imgstr = ''
        if pil_image.format == 'JPEG':
            imgstr = cv2.imencode('.jpg', patches[i])[1].tostring()
        elif pil_image.format == 'PNG':
            imgstr = cv2.imencode('.png', patches[i])[1].tostring()
        patches[i] = base64.b64encode(imgstr)
    return patches

def svhn(pil_image, province):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if province == 'nacao':
        img = nacao(img)
    imgstr = ''
    if pil_image.format == 'JPEG':
        imgstr = cv2.imencode('.jpg', img)[1].tostring()
    elif pil_image.format == 'PNG':
        imgstr = cv2.imencode('.png', img)[1].tostring()
    return base64.b64encode(imgstr)

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

def chq(img):
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    word_pix = np.argsort(-hist_r.astype(int), axis=0)[1][0]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if not(img[y][x][0] == word_pix and img[y][x][1] == word_pix and img[y][x][2] == word_pix):
                img[y][x][0] = 255
                img[y][x][1] = 255
                img[y][x][2] = 255
    alpha = cv2.cvtColor(img[10:40, 0:16], cv2.COLOR_BGR2GRAY)
    beta = cv2.cvtColor(img[10:40, 20:45], cv2.COLOR_BGR2GRAY)
    gamma = cv2.cvtColor(img[10:40, 44:57], cv2.COLOR_BGR2GRAY)
    alpha = uniformSize(norByMoments(alpha))
    beta = uniformSize(norByMoments(beta))
    gamma = uniformSize(norByMoments(gamma))
    return [alpha, beta, gamma]

def gs(img):
    alpha = img[25:43, 2:20]
    beta = img[25:43, 18:36]
    gamma = img[25:43, 36:54]
    return [alpha, beta, gamma]

def nx(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bBlur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(bBlur, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[8:27, 20:36]     # 19 * 16
    beta = thresh[12:27, 44:62]     # 15 * 18
    gamma = thresh[8:27, 70:85]     # 19 * 15
    return [uniformSize(alpha), uniformSize(beta), uniformSize(gamma)]

def tj(img):
    b, g, r = cv2.split(img)
    bBlur = cv2.bilateralFilter(g, 9, 75, 75)
    ret, thresh = cv2.threshold(bBlur, 160, 255, cv2.THRESH_BINARY)
    alpha = thresh[2:25, 10:25]     # 23 * 15
    beta = thresh[2:25, 42:62]      # 23 * 20
    gamma = thresh[2:25, 70:85]     # 23 * 15
    return [uniformSize(alpha), uniformSize(beta), uniformSize(gamma)]

def jx(img):
    bBlur = cv2.bilateralFilter(img, 9, 175, 175)
    ret, thresh = cv2.threshold(bBlur, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[3:33, 13:38]     # 30 * 25
    beta = thresh[0:35, 35:70]      # 35 * 35
    gamma = thresh[3:33, 65:90]     # 30 * 35
    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    beta = cv2.cvtColor(beta, cv2.COLOR_BGR2GRAY)
    gamma = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)
    return [alpha, beta, gamma]

def small(img):
    # small captcha, 80*30, shanxi, sichuan, xinjiang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    alpha = thresh[3:18, 2:13]      # 15 * 11
    beta = thresh[3:19, 18:34]      # 16 * 16
    gamma = thresh[3:18, 32:43]     # 15 * 11
    return [alpha, beta, gamma]

def nacao(img):
    r, g, b = cv2.split(img)
    blur = cv2.bilateralFilter(255 - b, 7, 35, 35)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return adaptive

def beijing(img):
    return [img]

def tess_reco(pil_image, province):
    if province == 'jiangsu':
        return tess_reco_js(pil_image)
    elif province == 'liaoning':
        return tess_reco_ln(pil_image)

def tess_reco_js(pil_image):
    # using tesseract for jiangsu
    ts = time()
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    print time() - ts
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    print time() - ts
    ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    print time() - ts
    result = pytesseract.image_to_string(Image.fromarray(thresh))
    print time() - ts
    result = result.strip().replace(' ', '').replace('.', '')
    info = {'accu':50, 'expr':result, 'valid':True, 'answer':result}
    return info

def tess_reco_ln(pil_image):
    # using tesseract for liaoning
    result = pytesseract.image_to_string(pil_image, lang='chi_sim')
    result = result.strip().replace(' ', '')
    expr = result
    if len(result) == 5:
        result = result.replace('乘', '*').replace('加', '+')
        if result[1] == '+':
            result = int(result[0]) + int(result[2])
        elif result[1] == '*':
            result = int(result[0]) * int(result[2])
    info = {'accu':60, 'expr':expr, 'valid':True, 'answer':result}
    return info
