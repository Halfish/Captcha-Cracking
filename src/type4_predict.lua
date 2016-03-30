require 'nn'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-province', 'chq', 'which type of CAPTCHA to recognize?')
cmd:option('-picpath', '', 'specific the picture path, empty means standard test')
cmd:text()
opt = cmd:parse(arg or {})

-- step 1: load model
print('loading model...')
local model_num_name = '../models/model_type4_' .. opt.province .. '_num.t7'
local model_symb_name = '../models/model_type4_' .. opt.province .. '_symb.t7'
local model_num = torch.load(model_num_name)
local model_symb = torch.load(model_symb_name)

function eval(picpath)
    -- step 2: split source picture into alpha, beta and gamma
    --      depending on file cutAndDump.py
    local command = 'python ./type4_cut.py ' .. opt.province .. ' -f single -p ' .. picpath 
    os.execute(command)

    -- step 3: read alpha, beta, and gamma
    local alpha = image.load('alpha.png')
    local beta = image.load('beta.png')
    local gamma = image.load('gamma.png')
    -- os.execute('rm alpha.png beta.png gamma.png')

    -- step 4: normalization is an essential step, keeping consistence with training process
    alpha = alpha - alpha:mean()
    beta = beta - beta:mean()
    gamma = gamma - gamma:mean()

    -- step 5: predicting
    local output1 = model_num:forward(alpha)
    local v1, i1 = output1:max(1)
    local output2 = model_symb:forward(beta)
    local v2, i2 = output2:max(1)
    local output3 = model_num:forward(gamma)
    local v3, i3 = output3:max(1)

    -- step 6: parse output
    local codec = {'+', '-', '*'}
    output = {}
    output[1] = {i1[1] - 1, math.exp(v1[1])}
    output[2] = {codec[i2[1]], math.exp(v2[1])}
    output[3] = {i3[1] - 1, math.exp(v3[1])}

    return output
end

function standard_test()
    print('predicting')
    for i = 1, 200 do
        local format = '.jpg'
        if opt.province == 'chq' then
            format = '.png'
        end
        local picpath = path.join('../testpic/type4/' .. opt.province, (5000+i-1) .. format)
        local output = eval(picpath)
        print(string.format('i = %d, pred = %s, avg_accu = %4f', i+5000-1, 
            output[1][1] .. ' ' .. output[2][1] .. ' ' .. output[3][1], 
            (output[1][2] + output[2][2] + output[3][2]) / 3))
    end
end

if opt.picpath == '' then
    standard_test()
else
    local output = eval(opt.picpath)
    print(string.format('pred = %s, avg_accu = %4f',
        output[1][1] .. ' ' .. output[2][1] .. ' ' .. output[3][1], 
        (output[1][2] + output[2][2] + output[3][2]) / 3))
end
