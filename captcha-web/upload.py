#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os.path

from tornado.options import define, options
define("port", default=3000, help="run on the given port", type=int)

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
            province = self.get_argument("province")
            types = self.get_argument("type")
            result = crack(meta, province, types)
            print result
            self.write(result)
        else:
            print self.request
            self.write('failed to upload')
    pass

provinces_1 = {'gansu':'gs', 'jiangxi':'jx', 'ningxia':'nx', 'tianjin':'tj', 'chongqing':'chq'}
provinces_2 = {'shan3xi':'small', 'sichuan':'small', 'xinjiang':'small'}
province_dict = dict(provinces_1, ** provinces_2)

def crack(meta, province, types):
    global province_dict
    if not province_dict.has_key(province):
        return 'wrong province name'
    # given a picture, return captcha breaking results in json format
    upload_path = os.path.join(os.path.dirname(__file__), 'static/files/')
    filename = 'captcha.' + meta['filename'].split('.')[1]
    filename = os.path.join(upload_path, filename)
    with open(filename, 'wb') as up:
        up.write(meta['body'])
    command = 'th type4_predict.lua -province ' + province_dict[province] + ' -picpath ' + filename
    output = os.popen(command)
    output = output.read().strip()
    return output[4:-4] # remove unknown symbol
pass

if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
