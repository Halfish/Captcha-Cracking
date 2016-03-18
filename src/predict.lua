-- read a file
require 'image'
require 'cutorch'
require 'cunn'
require 'nngraph'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-type', 1, 'Which type to predict?')
cmd:option('-model', '../models/model_mix_12.t7', 'Which model to use?')
cmd:option('-testdir', '../testpic/type1/', 'test directory')
cmd:option('-num', 100, 'number')
opt = cmd:parse(arg or {})

local decoder_util = require 'decoder'
local decoder
if opt.type == 1 then
    decoder = decoder_util.create('../synpic/codec_type1.txt', 8)
elseif opt.type == 2 then
    decoder = decoder_util.create('../synpic/codec_type2.txt', 5)
end

model = torch.load(opt.model)
model:evaluate()

file = io.open(path.join(opt.testdir, 'label.txt'), 'r')
local accuracy = 0.0
for i = 1, opt.num do
    local img = image.load(path.join(opt.testdir, i .. '.jpg'))
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
    end
    local real_label = decoder:str2label(str)
    -- print("prediction label = ", pred_label)
    -- print("really label = ", real_label)
    if decoder:compareLabel(pred_label, real_label) then
        accuracy = accuracy + 1
    end
    if i % (opt.num / 10) == 0 then
        print(string.format("i = %d,  \tpred = %s,\tlabel = %s", i, 
        decoder:label2str(pred_label), decoder:label2str(real_label)))
    end
end
accuracy = accuracy / opt.num * 100
print(string.format('accuracy is = %.2f%%', accuracy))
