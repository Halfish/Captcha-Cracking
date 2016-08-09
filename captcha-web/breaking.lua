local cjson = require 'cjson'
require 'nngraph'
require 'image'
require 'base64'

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

print('loading models from type1 to type10')
local model1 = torch.load('../models/model_type1.t7')
local model2 = torch.load('../models/model_type2.t7')
local model3 = torch.load('../models/model_type3.t7')
local model5 = torch.load('../models/model_type5.t7')
local model6 = torch.load('../models/model_type6.t7')
local model7 = torch.load('../models/model_type7.t7')
local model8 = torch.load('../models/model_type8.t7')
local model10 = torch.load('../models/model_type10.t7')
local model103 = torch.load('../models/nacao6w.t7')
model1:evaluate()
model2:evaluate()
model3:evaluate()
model5:evaluate()
model6:evaluate()
model7:evaluate()
model8:evaluate()
model10:evaluate()
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
local decoder7 = decoder_util.create('../trainpic/codec_type7.txt', 7)
local decoder8 = decoder_util.create('../trainpic/codec_type8.txt', 7)
local decoder10 = decoder_util.create('../trainpic/codec_type10.txt', 4)

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


function eval(imgs, province, model_num, model_symb)
    for i = 1, #imgs do
        imgs[i]:add(-imgs[i]:mean())
        imgs[i]:div(imgs[i]:std())
    end
    alpha, beta, gamma = unpack(imgs)
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

function type4_reco(imgs, province)
    local model_num = type4_models[province .. '_num']
    local model_symb = type4_models[province .. '_symb']
    local output = eval(imgs, province, model_num, model_symb)
    local expr = output[1][1] .. output[2][1] .. output[3][1]
    local result = strmath(output[1][1], output[2][1], output[3][1])
    local accu = (output[1][2] + output[2][2] + output[3][2]) / 3
    accu = math.floor(accu * 10000 + 0.5) / 100

    local ret = {expr=expr, answer=tostring(result), accu=accu, valid=true}
    return cjson.encode(ret)
end

function normalizeImage(img)
    if img:size()[1] == 1 then
        local img2 = torch.Tensor(3, img:size()[2], img:size()[3])
        img2[1] = img[1]
        img2[2] = img[1]
        img2[3] = img[1]
        img = img2
    end
    if opt.gpuid > 0 then
        img = img:cuda()
    end
    img = image.rgb2yuv(img)
    local channels = {'y', 'u', 'v'}
    for i, channel in ipairs(channels) do
        img[i]:add(-img[i]:mean())
        img[i]:div(img[i]:std())
    end
    return img
end

function specifyType(img)
    local size = img:size():totable()
    local imgtype = 0
    if size[1] == 3 and size[2] == 53 and size[3] == 160 then
        local output = model23:forward(img)
        local _, index = output:max(1)
        imgtype = index[1] + 1     -- return 2 or 3
    elseif size[1] == 3 and size[2] == 40 and size[3] == 180 then
        local output = model56:forward(img)
        local _, index = output:max(1)
        imgtype = index[1] + 4     -- return 2 or 3
    elseif size[1] == 3 and size[2] == 50 and size[3] == 200 then
        imgtype = 1
    elseif size[1] == 3 and size[2] == 40 and size[3] == 260 then
        imgtype = 7
    elseif size[1] == 3 and size[2] == 50 and size[3] == 300 then
        imgtype = 8
    elseif size[1] == 3 and size[2] == 27 and size[3] == 100 then
        imgtype = 103
    end
    return imgtype
end

function chooseModel(img)
    local captype = specifyType(img)
    local model, decoder
    if captype == 1 then
        model, decoder = model1, decoder1
    elseif captype == 2 then
        model, decoder = model2, decoder2
    elseif captype == 3 then
        model, decoder = model3, decoder3
    elseif captype == 5 then
        model, decoder = model5, decoder2
    elseif captype == 6 then
        model, decoder = model6, decoder4
    elseif captype == 7 then
        model, decoder = model7, decoder7
    elseif captype == 8 then
        model, decoder = model8, decoder8
    elseif captype == 103 then
        model, decoder = model103, decoder5
    end
    return model, decoder, captype
end

function svhn_reco(img)
    local img = normalizeImage(img)
    local model, decoder, captype = chooseModel(img)
    local output = model:forward(img)
    local pred_label = decoder:output2label(output)
    local expr = decoder:label2str(pred_label)
    local answer = ''
    if captype == 1 or captype == 2 or captype == 5 or captype == 7 or captype == 8 then
        answer = tostring(decoder:str2answer(expr))
    elseif captype == 3 or captype == 6 or captype == 103 then
        answer = expr
    end
    local jsonstring = {expr=expr, answer=answer, valid=true, accu=70}
    return cjson.encode(jsonstring)
end

function single_reco(img)
    --img = normalizeImage(img)
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

function get_validstr()
    local file = io.open('../trainpic/chisayings.txt')
    local str = file:read("*all")
    file:close()
    str = string.split(str, '\n')
    local validstr = {}
    for i = 1, #str do
        local label = decoder10:str2label(str[i])
        local flag = true
        for j = 1, 4 do if label[j + 1] == -1 then flag = false end end
        if flag then table.insert(validstr, str[i]) end
    end
    return validstr
end
local chisaying_type10 = get_validstr()

function hubei_reco(img)
    img = image.rgb2y(img)      -- intend to convert from RGB to gray
    local input = torch.Tensor(4, 1, 40, 22):typeAs(img)
    local start = {8, 46, 82, 120}
    for i = 1, 4 do
        sub = img[{{}, {1, 40}, {start[i] + 1, start[i] + 22}}]
        sub[sub:lt(200.0 / 255)] = 0
        sub[sub:ge(200.0 / 255)] = 1
        input[i] = sub - sub:mean()
    end
    if opt.gpuid > 0 then input = input:cuda() end
    local output = model10:forward(input)
    local k = 10
    local _, index = output:topk(k, 2, true) -- top 10 candidates, sorted reversely
    local best_score = 0
    local best_str = ''
    for i = 1, #chisaying_type10 do
        local label = decoder10:str2label(chisaying_type10[i])[{{2, 5}}]
        label = label:resize(4, 1):expand(4, k):cuda()
        local score = index:eq(label):sum()
        if score > best_score then
            best_score = score
            best_str = chisaying_type10[i]
        end
    end
    local jsonstring = {expr=best_str, answer=best_str, valid=true, accu=50}
    return cjson.encode(jsonstring)
end

function handle_message(message)
    local id = message['id']
    local which_model = message['type']
    local province = message['province']
    local format = message['format']
    local imgs = message['imgs']
    print(string.format("%d %s new message: %s, %s, %s, %d imgs", 
            id, os.date('%X %x'), which_model, province, format, #imgs))
    for i = 1, #imgs do
        local img = base64.decode(imgs[i])
        img = torch.ByteTensor(torch.ByteStorage():string(img))
        if format == 'JPEG' then
            img = image.decompressJPG(img)
        elseif format == 'PNG' then
            img = image.decompressPNG(img)
        else
            error(string.format('not jpeg or png, but %s image', format))
        end
        imgs[i] = img
    end

    local result = ''
    if which_model == 'svhn' then
        ok, result = pcall(svhn_reco, imgs[1])
    elseif which_model == 'type4' then
        if province == 'single' then
            ok, result = pcall(single_reco, imgs[1])
        elseif province == 'hubei' then
            ok, result = pcall(hubei_reco, imgs[1])
        else
            ok, result = pcall(type4_reco, imgs, province)
        end
    end
    if not ok then
        print(result)
        jsondict = {expr='', answer='', valid=false, accu=0}
        result = cjson.encode(jsondict)
    end
    return id, result
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
        local id, result = handle_message(message)
        client2:publish(id, result)
        print(string.format('%d %s -> %s\n', id, os.date('%X'), result))
    end
end
