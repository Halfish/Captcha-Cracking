--[[
-- th train.lua 
--      means using default settings, which is equal to:
--          th train.lua -datpath type4_chq_num.dat -model chq -type num
--      this will train a model named 
--]]--
require 'nn';
require 'optim';
require 'math';

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options:")
cmd:option('-gpuid', -1, 'which GPU to choose, -1 means using CPU')
cmd:option('-datpath', 'type4_chq_num.dat', 'directory of training data')
cmd:option('-splitrate', 0.7, 'split rate for training and validation')
cmd:option('-model', 'chq', 'which model to use? [chq, gs, nx, tj, jx, small, nacao, bj, hb]')
cmd:option('-type', 'num', 'which model to use? num or symb or single')
cmd:option('-learningRate', 0.1, 'learning rate for cnn model')
cmd:option('-batchsize', 128, 'batch size for training and evaluating')
cmd:option('-maxiters', 300, 'maximum iterations to train')
cmd:option('-savefreq', 50, 'save frequency')
cmd:text()
opt = cmd:parse(arg or {})

-- step 1: loading data
local fullset = torch.load(opt.datpath)
t_size = math.floor(fullset.size * opt.splitrate)
v_size = fullset.size - t_size
trainset = {
    size = t_size,
    data = fullset.data[{{1, t_size}}]:double(),
    label = fullset.label[{{1, t_size}}]
}
validset = {
    size = v_size,
    data = fullset.data[{{t_size+1, fullset.size}}]:double(),
    label = fullset.label[{{t_size+1, fullset.size}}]
}
print(trainset)
print(validset)
trainset.data = trainset.data - trainset.data:mean()
validset.data = validset.data - validset.data:mean()

if opt.gpuid > 0 then
    print('running on GPU')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    trainset.data = trainset.data:cuda()
    trainset.label = trainset.label:cuda()
    validset.data = validset.data:cuda()
    validset.label = validset.label:cuda()
end

-- step 2: building cnn model
if opt.model == 'chq' or opt.model == 'nx' or opt.model == 'tj' then
    channel = 1
    img_size_x = 30
    img_size_y = 30
elseif opt.model == 'gs' then
    channel = 3
    img_size_x = 18
    img_size_y = 18
elseif opt.model == 'jx' then
    channel = 1
    if opt.type == 'num' then
        img_size_x = 30
        img_size_y = 25
    elseif opt.type == 'symb' then
        img_size_x = 35
        img_size_y = 35
    end
elseif opt.model == 'small' then
    channel = 1
    if opt.type == 'num' then
        img_size_x = 15
        img_size_y = 11
    elseif opt.type == 'symb' then
        img_size_x = 16
        img_size_y = 16
    end
elseif opt.model == 'hb' then
    channel = 1
    img_size_x = 22
    img_size_y = 40
elseif opt.model == 'nacao' then
    channel = 3
    img_size_x = 30
    img_size_y = 20
elseif opt.model == 'bj' then
    channel = 3
    img_size_x = 41
    img_size_y = 34
end
print('input image size = ', channel, img_size_x, img_size_y)

if opt.type == 'num' then
    nclass = 10 -- 10 digits to classify, '[0-9]'
elseif opt.type == 'symb' then
    nclass = 3  -- 2 or 3 symbols to classify, '[+-*]' 
elseif opt.type == 'single' then
    decoder_util = require 'decoder'
    if opt.model == 'bj' then
        decoder = decoder_util.create('../trainpic/codec_type9.txt', 1)
    elseif opt.model == 'hb' then
        decoder = decoder_util.create('../trainpic/codec_type10.txt', 1)
    end
    nclass = decoder.label_size
end

local model_util = require 'type4_model'
local model = model_util.createType10()
print(model)

local w, dl_dw = model:getParameters()
print('get ' .. tostring(w:size()[1]) .. ' parameters')

if opt.gpuid > 0 then
    model = model:cuda()
end

-- step 3: training
criterion = nn.ClassNLLCriterion()
if opt.gpuid > 0 then
    criterion = criterion:cuda()
end

sgd_params = {
    learningRate = opt.learningRate,
    learningRateDecay = 1e-5,
    weightDecay = 1e-3,
    momentum = 1e-4
}
x, dl_dx = model:getParameters()

step = function()
    -- step function means traverse the whole training set
    local accu = 0
    local total_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    for t = 1, trainset.size, opt.batchsize do
        count = count + 1
        xlua.progress(count, math.ceil(trainset.size / opt.batchsize))
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + opt.batchsize - 1, trainset.size) - t + 1
        local inputs = torch.Tensor(size, channel, img_size_x, img_size_y)
        local targets = torch.IntTensor(size)
        if opt.gpuid > 0 then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        for i = 1,size do
            local input = trainset.data[shuffle[i+t-1]]
            local target = trainset.label[shuffle[i+t-1]]
            inputs[i] = input
            targets[i] = target
        end
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end

        _, fs = optim.sgd(feval, x, sgd_params)

        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        total_loss = total_loss + fs[1]

        -- accuracy for trainset
        local _, indices = torch.max(model.output, 2)
        accu = accu + indices:eq(targets):sum()
    end

    -- normalize loss
    return total_loss / count, accu / trainset.size
end

-- step 4: validate
-- evaluate the accuracy of a dataset, like validset or testset
eval = function()
    local accu = 0
    local count = 0
    local total_loss = 0

    for i = 1, validset.size, opt.batchsize do
        local size = math.min(i + opt.batchsize - 1, validset.size) - i + 1
        local inputs = validset.data[{{i,i+size-1}}]
        local targets = validset.label[{{i,i+size-1}}]
        if opt.gpuid > 0 then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        local outputs = model:forward(inputs)
        local loss = criterion:forward(outputs, targets)
        local _, indices = torch.max(outputs, 2)
        local guessed_right = indices:eq(targets):sum()
        accu = accu + guessed_right
        count = count + 1
        total_loss = total_loss + loss
    end

    return total_loss / count, accu / validset.size
end

-- do start the training process
do
    local stoppingLR = opt.learningRate * 0.005
    local stopwatch = 0
    local last_v_loss = 100
    model_name = 'model_type4_' .. opt.model .. '_' .. opt.type ..'.t7'

    for i = 1, opt.maxiters do
        local timer = torch.Timer()
        model:training()
        local loss, accu = step()
        model:evaluate()
        local valid_loss, valid_accu = eval()
        print(string.format('Epoch: %d loss = %.3f,  accu=%.3f,  v_loss = %.3f,  v_accu = %.3f, costed %.3f s', 
            i, loss, accu, valid_loss, valid_accu, timer:time().real))

        if i % opt.savefreq == 0 then
            print('saving ' .. model_name)
            torch.save(model_name, model)
        end

        if valid_loss > last_v_loss then
            if stopwatch >= 6 then
                if sgd_params.learningRate < stoppingLR then
                    break
                else
                    sgd_params.learningRate = sgd_params.learningRate / 2
                    stopwatch = 0
                    print(string.format('new learning rate is %f', sgd_params.learningRate))
                end
            else
                stopwatch = stopwatch + 1
            end
        end
        last_v_loss = valid_loss
    end
end
