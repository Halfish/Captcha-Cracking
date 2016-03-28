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
cmd:option('-datpath', 'type4_chq_num.dat', 'directory of training data')
cmd:option('-splitrate', 0.7, 'split rate for training and validation')
cmd:option('-model', 'chq', 'which model to use? [chq, gs, nx, tj, jx]')
cmd:option('-type', 'num', 'which model to use? num or symb')
cmd:option('-maxiters', 300, 'maximum iterations to train')
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
print(trainset.size)
print(validset.size)
trainset.data = trainset.data - trainset.data:mean()
validset.data = validset.data - validset.data:mean()

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
end
print('input image size = ', channel, img_size_x, img_size_y)

if opt.type == 'num' then
    nclass = 10 -- 10 digits to classify, '[0-9]'
elseif opt.type == 'symb' then
    nclass = 3  -- 2 or 3 symbols to classify, '[+-*]' 
end

kernel_num = 10
kernel_size = 5
model = nn.Sequential()
model:add(nn.View(channel, img_size_x, img_size_y))
model:add(nn.SpatialConvolutionMM(channel, kernel_num, kernel_size, kernel_size))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
local num1 = math.floor((img_size_x - 4) / 2)
local num2 = math.floor((img_size_y - 4) / 2)
local num3 = kernel_num * num1 * num2
model:add(nn.View(num3))
model:add(nn.Linear(num3, nclass * 10))
model:add(nn.ReLU())
model:add(nn.Linear(nclass * 10, nclass))
model:add(nn.LogSoftMax())
model = require('weight-init')(model, 'xavier')

-- step 3: training
criterion = nn.ClassNLLCriterion()
sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-5,
    weightDecay = 1e-3,
    momentum = 1e-4
}
x, dl_dx = model:getParameters()

step = function(batch_size)
    -- step function means traverse the whole training set
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    for t = 1, trainset.size, batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t 
        local inputs = torch.Tensor(size, channel, img_size_x, img_size_y)--:cuda()
        --local inputs = torch.Tensor(size, 30, 30)--:cuda()
        local targets = torch.Tensor(size)--:cuda()
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            -- if target == 0 then target = 10 end ?
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
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
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

-- step 4: validate
-- evaluate the accuracy of a dataset, like validset or testset
eval = function(validset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
    for i = 1, validset.size, batch_size do
        local size = math.min(i + batch_size - 1, validset.size) - i
        local inputs = validset.data[{{i,i+size-1}}]--:cuda()
        local targets = validset.label[{{i,i+size-1}}]:long()--:cuda()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / validset.size
end

-- do start the training process
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1, opt.maxiters do
        local loss = step()
        local accuracy = eval(validset)
        if i % 10 == 0 then
            print(string.format('Epoch: %d loss = %4f\t valid_accu = %4f', i, loss, accuracy))
        end
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

-- step 5: saving the model
model_name = 'model_type4_' .. opt.model .. '_' .. opt.type ..'.t7'
print('saving ' .. model_name)
torch.save(model_name, model)

-- testset.data = testset.data:double()
-- eval(testset)
