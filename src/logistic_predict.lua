cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-type', 23, 'Which type to predict?')
cmd:option('-model', 'model.t7', 'Which model to use?')
cmd:option('-testdir', '../testpic/type2/', 'test directory')
cmd:option('-num', 100, 'number')

opt = cmd:parse(arg or {})

require 'image'
require 'nngraph'

local decoder_util = require 'decoder'
local decoder

print('loading model...')
model = torch.load(opt.model)
model:evaluate()
print('model loaded')

local format = ''
if opt.type == 23 then
    format = '.jpg'
elseif opt.type == 56 then
    format = '.png'
end

local accuracy = 0.0
for i = 1, opt.num do
    local img = image.load(path.join(opt.testdir, i .. format))
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
    local output = model:forward(img)
    local _, indices = output:max(1)
    print(indices[1])
    if indices[1] == 1 then
        accuracy = accuracy + 1
    end
end
accuracy = accuracy / opt.num * 100
print(string.format('accuracy is = %.2f%%', accuracy))
