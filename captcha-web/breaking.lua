local cjson = require 'cjson'
require 'nngraph'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-gpuid', -1, 'GPU id')
opt = cmd:parse(arg or {})

if opt.gpuid > 0 then
    print('Loading GPU...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
end

print('loading models from type1 to type9')
local model1 = torch.load('../models/model_type1.t7')
local model2 = torch.load('../models/model_type2.t7')
local model3 = torch.load('../models/model_type3.t7')
local model5 = torch.load('../models/model_type5.t7')
local model6 = torch.load('../models/model_type6.t7')
local model103 = torch.load('../models/nacao6w.t7')
model1:evaluate()
model2:evaluate()
model3:evaluate()
model5:evaluate()
model6:evaluate()
model103:evaluate()

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
local decoder4 = decoder_util.create('../trainpic/codec_type6.txt', 4)
local decoder5 = decoder_util.create('../trainpic/codec_nacao.txt', 6)

-- loading type4 models
local type4_provinces = {'gs', 'jx', 'nx', 'tj', 'chq', 'small', 'nacao'}
local type4_models = {}
print('loading type4 models')
for i, p in ipairs(type4_provinces) do
    local model = torch.load('../models/model_type4_' .. p .. '_num.t7')
    type4_models[p .. '_num'] = model
    model = torch.load('../models/model_type4_' .. p .. '_symb.t7')
    type4_models[p .. '_symb'] = model
end

print('loading beijing model')
local bjModel = torch.load('../models/model_type4_bj_single.t7')
local bjDecoder = decoder_util.create('../trainpic/codec_type9.txt', 1)


function eval(filename, province, model_num, model_symb)
    local alpha = image.load('alpha.png')
    local beta = image.load('beta.png')
    local gamma = image.load('gamma.png')
    alpha = alpha - alpha:mean()
    beta = beta - beta:mean()
    gamma = gamma - gamma:mean()

    local output1 = model_num:forward(alpha)
    local v1, i1 = output1:max(1)
    local output2 = model_symb:forward(beta)
    local v2, i2 = output2:max(1)
    local output3 = model_num:forward(gamma)
    local v3, i3 = output3:max(1)

    local codec = {'+', '-', '*'}
    output = {}
    output[1] = {i1[1] - 1, math.exp(v1[1])}
    output[2] = {codec[i2[1]], math.exp(v2[1])}
    output[3] = {i3[1] - 1, math.exp(v3[1])}

    return output
end

function strmath(a, b, c)
    if b == '+' then
        return a + c
    elseif b == '-' then
        return a - c
    elseif b == '*' then
        return a * c
    end
end

function type4_reco(filename, province)
    local model_num = type4_models[province .. '_num']
    local model_symb = type4_models[province .. '_symb']
    local output = eval(filename, province, model_num, model_symb)
    local expr = output[1][1] .. output[2][1] .. output[3][1]
    local result = strmath(output[1][1], output[2][1], output[3][1])
    local accu = (output[1][2] + output[2][2] + output[3][2]) / 3
    accu = math.floor(accu * 10000 + 0.5) / 100

    local ret = {expr=expr, answer=tostring(result), accu=accu, valid=true}
    return cjson.encode(ret)
end


function readImage(filename)
    local img = image.load(filename, 3)
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
    elseif size[1] == 3 and size[2] == 50 and size[3] == 200 then
        return 1
    elseif size[1] == 3 and size[2] == 27 and size[3] == 100 then
        return 103
    end
end

function chooseModel(img)
    local captype = specifyType(img)
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
        decoder = decoder4
    elseif captype == 103 then
        model = model103
        decoder = decoder5
    end
    return model, decoder, captype
end

function svhn_reco(filename)
    local img = readImage(filename)
    local model, decoder, captype = chooseModel(img)
    local output = model:forward(img)
    local pred_label = decoder:output2label(output)
    local expr = decoder:label2str(pred_label)
    local answer = ''
    if captype == 1 or captype == 2 or captype == 5 then
        answer = tostring(decoder:str2answer(expr))
    elseif captype == 3 or captype == 6 or captype == 103 then
        answer = expr
    end
    local jsonstring = {expr=expr, answer=answer, valid=true, accu=70}
    return cjson.encode(jsonstring)
end

function single_reco(filename)
    local img = image.load(filename)
    local alpha = {}
    alpha[1] = img[{{}, {8, 48}, {28, 61}}]
    alpha[2] = img[{{}, {8, 48}, {58, 91}}]
    alpha[3] = img[{{}, {8, 48}, {90, 123}}]
    alpha[4] = img[{{}, {8, 48}, {117, 150}}]

    local label = ''
    for i = 1, 4 do
        local _, index = bjModel:forward(alpha[i] - alpha[i]:mean()):max(1)
        label = label .. bjDecoder.rev_mapper[index[1]]
    end
    local jsonstring = {expr=label, answer=label, valid=true, accu=50}
    return cjson.encode(jsonstring)
end

-- see https://github.com/nrk/redis-lua/blob/version-2.0/examples/pubsub.lua 
local redis = require 'redis'
local client = redis.connect('127.0.0.1', 6379)
local client2 = redis.connect('127.0.0.1', 6379)

for msg in client:pubsub({subscribe = {'request'}}) do
    if msg.kind == 'subscribe' then
        print('subscribe to channel ' .. msg.channel)
        print('Lua -> I am ready!\n')
    elseif msg.kind == 'message' then
        message = cjson.decode(msg.payload)
        print('Lua: Oh I have got job to do.:')
        print(message, '\n')
        local id = message['id']
        local province = message['province']
        local filename = message['filename']
        local which_model = message['type']
        local result = ''
        if which_model == 'svhn' then
            ok, result = pcall(svhn_reco, filename)
        elseif which_model == 'type4' then
            ok, result = pcall(type4_reco, filename, province)
        elseif which_model == 'single' then
            ok, result = pcall(single_reco, filename)
        end
        if not ok then
            jsondict = {expr='', answer='', valid=false, accu=0}
            result = cjson.encode(jsondict)
        end
        client2:publish(id, result)
        print('answer to', id, 'is', result)
    end
end
