cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-gpuid', -1, 'GPU id')
opt = cmd:parse(arg or {})

require 'nngraph'
require 'image'
if opt.gpuid > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
end

local model1 = torch.load('../models/model_type1.t7')
local model2 = torch.load('../models/model_type2.t7')
local model3 = torch.load('../models/model_type3.t7')
local model5 = torch.load('../models/model_type5.t7')
local model6 = torch.load('../models/model_type6.t7')
local model9 = torch.load('../models/model_type9.t7')
model1:evaluate()
model2:evaluate()
model3:evaluate()
model5:evaluate()
model6:evaluate()
model9:evaluate()

local model23 = torch.load('../models/model_log_type23.t7')
local model56 = torch.load('../models/model_log_type56.t7')
model23:evaluate()
model56:evaluate()

if opt.gpuid > 0 then
    model23 = model23:cuda()
    model56 = model56:cuda()
end

local decoder_util = require 'decoder'
local decoder1 = decoder_util.create('../trainpic/codec_type1.txt', 8)
local decoder2 = decoder_util.create('../trainpic/codec_type2.txt', 5)
local decoder3 = decoder_util.create('../trainpic/chisayings.txt', 4)
local decoder4 = decoder_util.create('../trainpic/codec_type9.txt', 4)

function readImage(filename)
    local img = image.load(filename)
    img = image.rgb2yuv(img)
    local channels = {'y', 'u', 'v'}
    local mean = {}
    local std = {}
    for i, channel in ipairs(channels) do
        mean[i] = img[i]:mean()
        std[i] = img[i]:std()
        img[i]:add(-mean[i])
        img[i]:div(std[i])
    end
    if opt.gpuid > 0 then
        img = img:cuda()
    end
    return img
end

function specifyType(img)
    local size = img:size()
    if size[1] == 3 and size[2] == 53 and size[3] == 160 then
        local output = model23:forward(img)
        local _, index = output:max(1)
        print('type', index[1]+1, 'specified')
        return index[1] + 1     -- return 2 or 3
    elseif size[1] == 3 and size[2] == 40 and size[3] == 180 then
        local output = model56:forward(img)
        local _, index = output:max(1)
        print('type', index[1]+4, 'specified')
        return index[1] + 4     -- return 2 or 3
    end
end

function chooseModel(img, captype)
    if captype == 0 then
        captype = specifyType(img)
    end
    local model, decoder
    if captype == 1 then
        model = model1
        decoder = decoder1
    elseif captype == 2 then
        model = model2
        decoder = decoder2
    elseif captype == 3 then
        model = model3
        decoder = decoder3
    elseif captype == 5 then
        model = model5
        decoder = decoder2
    elseif captype == 6 then
        model = model6
        decoder = decoder3
    elseif captype == 9 then
        model = model9
        decoder = decoder4
    end
    return model, decoder
end

function crack(filename, captype)
    print(filename)
    local img = readImage(filename)
    local model, decoder = chooseModel(img, captype)
    local output = model:forward(img)
    local pred_label = decoder:output2label(output)
    return decoder:label2str(pred_label)
end

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
        local id = message['id']
        local province = message['province']
        local filename = message['filename']
        local captype = message['type']
        local result = crack(filename, captype)
        client2:publish(id, result)
    end
end
