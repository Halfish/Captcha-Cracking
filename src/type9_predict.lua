require 'nn'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-picpath', '', 'specific the picture path, empty means standard test')
cmd:text()
opt = cmd:parse(arg or {})

-- step 1: load model
-- print('loading model...')
local model = torch.load('../models/model_type4_bj_single.t7')
local decoder_util = require 'decoder'
local decoder = decoder_util.create('../trainpic/codec_type9.txt')

function eval(picpath)
    local img = image.load(picpath)
    local alphabeta = {}
    alphabeta[1] = img[{{}, {8, 48}, {28, 61}}]
    alphabeta[2] = img[{{}, {8, 48}, {58, 91}}]
    alphabeta[3] = img[{{}, {8, 48}, {90, 123}}]
    alphabeta[4] = img[{{}, {8, 48}, {117, 150}}]

    -- step 4: normalization is an essential step, keeping consistence with training process
    local label = ''
    for i = 1, 4 do
        local img = alphabeta[i] - alphabeta[i]:mean()
        local output = model:forward(img)
        local v, i = output:max(1)
        label = label .. decoder.rev_mapper[i[1]]
    end
    return label 
end

function standard_test()
    local accu = 0
    local num = 99
    local file = io.open('../testpic/type9/label.txt', 'r')
    print('predicting')
    for i = 1, num do
        local picpath = '../testpic/type9/' .. i .. '.jpg'
        local output = eval(picpath)
        local label = file:read()
        if output == label then
            accu = accu + 1
        end
        print(string.format('i = %d, pred = %s, label = %s, accu = %d', i, output, label, accu))
    end
end

standard_test()
