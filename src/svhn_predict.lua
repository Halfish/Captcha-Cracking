cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-type', 1, 'Which type to predict?')
cmd:option('-model', '../models/model_mix_12.t7', 'Which model to use?')
cmd:option('-testdir', '../testpic/type1/', 'test directory')
cmd:option('-num', 100, 'number')
opt = cmd:parse(arg or {})

require 'image'
require 'cutorch'
require 'cunn'
require 'nngraph'

local decoder_util = require 'decoder'
local decoder
if opt.type == 1 then
    decoder = decoder_util.create('../trainpic/codec_type1.txt', 8)
elseif opt.type == 2 then
    decoder = decoder_util.create('../trainpic/codec_type2.txt', 5)
elseif opt.type == 3 then
    decoder = decoder_util.create('../trainpic/chisayings.txt', 4)
elseif opt.type == 9 then
    decoder = decoder_util.create('../trainpic/codec_type9.txt', 4)
end

model = torch.load(opt.model)
model:evaluate()

file = io.open(path.join(opt.testdir, 'label.txt'), 'r')
local accuracy = 0.0
for i = 1, opt.num do
    local img = image.load(path.join(opt.testdir, i .. '.jpg'), 1)
    if opt.type == 9 then
        k = image.gaussian(3)
        img = image.convolve(img, k, 'same')
        img[img:lt(0.99)] = 0
        img[img:ge(0.99)] = 1
        img2 = image.load(path.join(opt.testdir, i .. '.jpg'), 3)
        img2[1] = img
        img2[2] = img
        img2[3] = img
        img = img2
    end
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
    img = img:cuda()
    local output = model:forward(img)
    local pred_label = decoder:output2label(output)
    local str = ''
    if opt.type == 1 then
        str = decoder:simple2str_type1(file:read())
    elseif opt.type == 2 then
        str = decoder:simple2str_type2(file:read())
    elseif opt.type == 3 or opt.type == 9 then
        str = file:read()
    end
    local real_label = decoder:str2label(str)
    -- print("prediction label = ", pred_label)
    -- print("really label = ", real_label)
    if decoder:compareLabel(pred_label, real_label) then
        accuracy = accuracy + 1
    end
    if i % 5 == 0 then
        print(string.format("i = %d,  \tpred = %s,\tlabel = %s", i, 
        decoder:label2str(pred_label), decoder:label2str(real_label)))
    end
end
accuracy = accuracy / opt.num * 100
print(string.format('accuracy is = %.2f%%', accuracy))
