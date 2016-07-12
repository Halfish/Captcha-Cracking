#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os
import redis
import json
import random
from tornado import gen
from tornado.concurrent import run_on_executor, futures
from tornado.ioloop import IOLoop
import categorize

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

client = redis.StrictRedis(host='localhost', port=6379, db=0, socket_timeout=3)

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
    @gen.coroutine
    def get(self):
        self.render(
            "index.html",
        )
    pass

    @gen.coroutine
    def post(self):
        if len(self.request.files) > 0:
            imgfile = self.request.files['file'][0]['body']
            province = self.get_argument("province")
            print 'province:' + province
            result = yield do_stuff(imgfile, province)
            self.write(result)
            print json.loads(result)['expr']
        else:
            print self.request
            self.write('failed to upload')
    pass

class TaskRunner(object):
    def __init__(self, loop=None):
        self.executor = futures.ThreadPoolExecutor(4)
        self.loop = loop or IOLoop.instance()

    @run_on_executor
    def long_running_task(self, imgfile, province):
        captype, capformat, province, capinfo = categorize.crack(imgfile, province)
        if captype == 'tess':
            return json.dumps(capinfo)
        elif captype in ['svhn', 'type4']:
            return broadcast(province, captype, capformat, capinfo)
        else:
            # no such province
            info = {'accu':0, 'expr':'', 'valid':False, 'answer':'', 'notes':'No such province, hiahia'}
            return json.dumps(info)

tasks = TaskRunner()
@gen.coroutine
def do_stuff(imgfile, province):
    result = yield tasks.long_running_task(imgfile, province)
    raise gen.Return(result)

def broadcast(province, captype, capformat, capimg):
    capid = random.randint(100000000, 999999999)
    info = {'id':capid, 'province':province, 'type':captype, 'format':capformat, 'imgs':capimg}
    p = client.pubsub()
    p.subscribe(str(capid)) # must subscribe first in case to not miss the message
    client.publish('request', json.dumps(info))
    result = ''
    for item in p.listen():
        if item['type'] == 'subscribe':
            #print 'python:subscribed ' + str(id) + ' successfully.'
            pass
        elif item['type'] == 'message':
            result = item['data']
            p.unsubscribe(str(capid))
            p.close()
            break
        pass
    return result

if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
