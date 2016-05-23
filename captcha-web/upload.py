#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os
import redis
import json
import random
import cv2
import type4_cut

client = redis.StrictRedis(host='localhost', port=6379, db=0, socket_timeout=3)
p = client.pubsub()

from tornado.options import define, options
define("port", default=3001, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", UploadFileHandler),
        ]
        settings = dict(
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            debug = True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(
            "index.html",
        )
    pass

    def post(self):
        if len(self.request.files) > 0:
            meta = self.request.files['file'][0]
            upload_path = os.path.join(os.path.dirname(__file__), 'static/files/')
            #filename = 'captcha.' + meta['filename'].split('.')[1]
            filename = 'captcha'
            filename = os.path.join(upload_path, filename)
            with open(filename, 'wb') as up:
                up.write(meta['body'])
            pass

            province = self.get_argument("province")
            self.write(crack(filename, province))
            #self.write('<p>^_^</p>')
        else:
            print self.request
            self.write('failed to upload')
    pass

provinces_1 = {'gansu':'gs', 'jiangxi':'jx', 'ningxia':'nx', 'tianjin':'tj', 'chongqing':'chq', 'nacao':'nacao'}
provinces_2 = {'shan3xi':'small', 'sichuan':'small', 'xinjiang':'small'}
type4_province_dict = dict(provinces_1, ** provinces_2)
svhn_provinces = ['anhui', 'guangxi', 'henan', 'heilongjiang', 'qinghai', 'shanxi',
                'xizang', 'fujian', 'nation', 'hebei', 'shanghai', 'yunnan', 'hunan',
                  'guangdong', 'hainan', 'neimenggu', 'nacao']
tess_provinces = ['jiangsu', 'liaoning']
prep_mapper = {'chq':type4_cut.preprocess_chq, 'gs':type4_cut.preprocess_gs,
        'nx':type4_cut.preprocess_nx, 'tj':type4_cut.preprocess_tj,
        'jx':type4_cut.preprocess_jx, 'small':type4_cut.preprocess_small}
single_provinces = ['beijing']

def crack(filename, province):
    global province_dict
    global svhn_provinces
    if type4_province_dict.has_key(province) or province in svhn_provinces or province in single_provinces:
        id = random.randint(100000, 999999)
        which_model = ''
        if province in svhn_provinces:
            which_model = 'svhn'
            if province == 'nacao':
                type4_cut.nacao3(filename)
        elif province in single_provinces:
            which_model = 'single'
        else:
            which_model = 'type4'
            province = type4_province_dict[province]
            f = prep_mapper[province]   # preprocess
            f(filename, 'alpha.png', 'beta.png', 'gamma.png')
        pass
        info = {'type':which_model, 'filename':filename, 'id':id, 'province':province}
        p.subscribe(str(id)) # must subscribe first in case to not miss the message
        client.publish('request', json.dumps(info))
        result = ''
        for item in p.listen():
            if item['type'] == 'subscribe':
                print 'python:subscribed ' + str(id) + ' successfully.'
            elif item['type'] == 'message':
                print('python', result)
                result = item['data']
                break
            pass
        return result
    elif province in tess_provinces:
        print 'type0, using tesseract'
        if province == 'jiangsu':
            return tess_reco_js(filename)
        elif province == 'liaoning':
            return tess_reco_ln(filename)
        pass
    pass

    # no such province
    info = {'accu':0, 'expr':'', 'valid':False, 'result':'', 'notes':'No such province, hiahia'}
    return json.dumps(info)

import pytesseract
import Image
def tess_reco_js(filename):
    img = cv2.imread(filename, 0)
    blur = cv2.bilateralFilter(img, 5, 75, 75)
    ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    cv2.imwrite('binary.jpg', thresh)
    result = pytesseract.image_to_string(Image.open('binary.jpg'))
    result = result.strip().replace(' ', '').replace('.', '')
    info = {'accu':50, 'expr':result, 'valid':True, 'result':result}
    return json.dumps(info)
pass

def tess_reco_ln(filename):
    result = pytesseract.image_to_string(Image.open(filename), lang='chi_sim')
    result = result.strip().replace(' ', '')
    print 'captcha is: ' + result
    expr = result
    if len(result) == 5:
        result = result.replace('乘', '*').replace('加', '+')
        if result[1] == '+':
            result = int(result[0]) + int(result[2])
        elif result[1] == '*':
            result = int(result[0]) * int(result[2])
    info = {'accu':60, 'expr':expr, 'valid':True, 'result':result}
    return json.dumps(info)


if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
