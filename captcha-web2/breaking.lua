-- see https://github.com/nrk/redis-lua/blob/version-2.0/examples/pubsub.lua 

local redis = require 'redis'
local cjson = require 'cjson'

local client = redis.connect('127.0.0.1', 6379)
local client2 = redis.connect('127.0.0.1', 6379)

for msg in client:pubsub({subscribe = {'request'}}) do
    if msg.kind == 'subscribe' then
        print('subscribe to channel ' .. msg.channel)
    elseif msg.kind == 'message' then
        message = cjson.decode(msg.payload)
        print('message:', message)

        id = message['id']
        filename = message['filename']
        client2:publish(id, id .. ' ' .. filename)
    end
end
