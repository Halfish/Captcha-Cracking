#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os.path
import redis
import json

client = redis.StrictRedis(host='localhost', port=6379, db=0)
p = client.pubsub()

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
            type = self.get_argument("type")
            self.write(crack(meta, province, type))
            self.write('<p>^_^</p>')
        else:
            print self.request
            self.write('failed to upload')
    pass


def crack(meta, province, type):
    global province_dict
    # given a picture, return captcha breaking results in json format
    upload_path = os.path.join(os.path.dirname(__file__), 'static/files/')  
    filename = 'captcha.' + meta['filename'].split('.')[1]
    filename = os.path.join(upload_path, filename)
    with open(filename, 'wb') as up:      
        up.write(meta['body'])
    id = 123456
    #info = {'filename':filename, 'id':id}
    info = {'filename':meta['filename'], 'id':id}
    client.publish('request', json.dumps(info))
    p.subscribe(str(id))
    result = ''
    for item in p.listen():
        if item['type'] == 'subscribe':
            print 'subscribed ' + str(id) + ' successfully.'
        elif item['type'] == 'message':
            result = item['data']
            break
    return result


if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
