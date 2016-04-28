require 'nn';
require 'math';
require 'optim';

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options:")
cmd:option('-gpuid', -1, 'which GPU to choose, -1 means using CPU')
cmd:option('-type', 23, 'which type to choose, 23 or 56?')
cmd:option('-datapath', 'fullset.dat', 'directory of training data')
cmd:option('-splitrate', 0.7, 'split rate for training and validation')
cmd:option('-learningRate', 0.1, 'learning rate for cnn model')
cmd:option('-maxiters', 300, 'maximum iterations to train')
cmd:option('-savefreq', 10, 'save frequency')
cmd:option('-savename', 'model.t7', 'save name')
cmd:text()
opt = cmd:parse(arg or {})

-- step 1: loading data
local fullset = torch.load(opt.datapath)
-- mnist = require 'mnist'
-- fullset = mnist.traindataset()

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
print(trainset.size)
print(validset.size)
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
local channel, img_size_x, img_size_y
if opt.type == 23 then
    channel = 3 
    img_size_x = 53
    img_size_y = 160
elseif opt.type == 56 then
    channel = 3 
    img_size_x = 40
    img_size_y = 180
end

local model_util = require 'type4_model'
local model_config = {
    picsize = {channel, img_size_x, img_size_y},
    n_conv_layers = 2,
    filter_num = {channel, 4, 8},
    filter_size = 5,
    dropout_value = 0.5,
    n_full_connect = 10,
    nclass = 2 
}
model = model_util.create(model_config)

print(model)

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
print('get', (#x)[1], 'parameters')

step = function()
    -- step function means traverse the whole training set
    local accu = 0
    local total_loss = 0
    local shuffle = torch.randperm(trainset.size)
    for i = 1, trainset.size do
        local data = trainset.data[shuffle[i]]
        local label = trainset.label[shuffle[i]]
        if opt.gpuid > 0 then
            data = data:cuda()
            label = label:cuda()
        end
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()
            local loss = criterion:forward(model:forward(data), label)
            model:backward(data, criterion:backward(model.output, label))
            return loss, dl_dx
        end

        _, fs = optim.sgd(feval, x, sgd_params)

        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        total_loss = total_loss + fs[1]

        -- accuracy for trainset
        local _, index = model.output:max(1)
        index = index[1]
        if index == label then
            accu = accu + 1
        end
    end

    -- normalize loss
    return total_loss / trainset.size, accu / trainset.size
end

-- step 4: validate
-- evaluate the accuracy of a dataset, like validset or testset
eval = function(validset, batch_size)
    local accu = 0
    local total_loss = 0
    
    for i = 1, validset.size do
        local data = validset.data[i]
        local label = validset.label[i]
        if opt.gpuid > 0 then
            data = data:cuda()
            label = label:cuda()
        end
        local output = model:forward(data)
        local loss = criterion:forward(output, label)

        if i == 10 then
            -- local last = model:get(2)
            -- print(last.output)
        end

        local _, index = model.output:max(1)
        if index[1] == label then
            accu = accu + 1
        end
        total_loss = total_loss + loss
    end

    return total_loss / validset.size, accu / validset.size
end

-- do start the training process
do
    local stoppingLR = opt.learningRate * 0.005
    local stopwatch = 0
    local last_v_loss = 100

    for i = 1, opt.maxiters do
        local timer = torch.Timer()
        model:training()
        local loss, accu = step()
        model:evaluate()
        local valid_loss, valid_accu = eval(validset)
        print(string.format('Epoch: %d loss = %.3f,  accu=%.3f,  v_loss = %.3f,  v_accu = %.3f, costed %.3f s', 
            i, loss, accu, valid_loss, valid_accu, timer:time().real))

        if i % opt.savefreq == 0 then
            torch.save(opt.savename, model)
            print('saving ' .. opt.savename)
        end

        if valid_loss > last_v_loss then
            if stopwatch >= 3 then
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
