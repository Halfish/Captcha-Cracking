#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os
import redis
import json
import random

client = redis.StrictRedis(host='localhost', port=6379, db=0)
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
            filename ='captcha.jpg'
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

provinces_1 = {'gansu':'gs', 'jiangxi':'jx', 'ningxia':'nx', 'tianjin':'tj', 'chongqing':'chq'}
provinces_2 = {'shan3xi':'small', 'sichuan':'small', 'xinjiang':'small'}
type4_province_dict = dict(provinces_1, ** provinces_2)
svhn_provinces = ['anhui', 'guangxi', 'henan', 'heilongjiang', 'qinghai', 'shanxi',
                'xizang', 'fujian', 'nation', 'hebei', 'shanghai', 'yunnan', 'hunan',
                  'guangdong', 'hainan', 'neimenggu']

def crack(filename, province):
    global province_dict
    global svhn_provinces
    if type4_province_dict.has_key(province):
        print 'python:using type4 model'
        command = 'th type4_predict.lua -province ' + type4_province_dict[province] + ' -picpath ' + filename
        output = os.popen(command)
        output = output.read().strip()
        return output # remove unknown symbol
    elif province in svhn_provinces:
        print 'python:using svhn model'
        id = random.randint(100000, 999999)
        info = {'filename':filename, 'id':id, 'province':province}
        client.publish('request', json.dumps(info))
        p.subscribe(str(id))
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
    pass

    # no such province
    info = {'accu':0, 'expr':'', 'valid':False, 'result':'', 'notes':'No such province, hiahia'}
    return json.dumps(info)

if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
