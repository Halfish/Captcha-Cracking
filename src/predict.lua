-- read a file
require 'decoder'
require 'image'
require 'cutorch'
require 'cunn'
require 'nngraph'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-model', '../models/model_mix_12.t7', 'which model to use')
cmd:option('-testdir', '../testpic/', 'test directory')
cmd:option('-num', 100, 'number')
opt = cmd:parse(arg or {})

model = torch.load(opt.model)
model:evaluate()

file = io.open(opt.testdir .. 'label.txt', 'r')
local accuracy = 0.0
for i = 1, opt.num do
    local img = image.load(opt.testdir .. (i+1900-1) .. '.jpg')
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
    local pred_label = output2label(output)
    local real_label = str2label(simple2str(file:read()))
    -- print("prediction label = ", pred_label)
    -- print("really label = ", real_label)
    if compareLabel(pred_label, real_label) then
        accuracy = accuracy + 1
    end
    if i % (opt.num / 10) == 0 then
        print(string.format("i = %d,  \tpred = %s,\tlabel = %s", i, label2str(pred_label), label2str(real_label)))
    end
end
accuracy = accuracy / opt.num * 100
print(string.format('accuracy is = %.2f%%', accuracy))
