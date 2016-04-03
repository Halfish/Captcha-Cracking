require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options:")

-- global:
cmd:option('-gpuid', -1, 'id of gpu, negative values like -1 means using CPU')
-- cmd:option('-threads', 2, 'number of threads')
cmd:option('-model', '', 'using trained model')
cmd:option('-type', 1, 'which type of CAPTCHA to train')
cmd:option('-dropout', 0.5, 'prob of dropout for model')
cmd:option('-savename', 'model.t7', 'the name of model to save')
cmd:option('-dataname', 'fullset.dat', 'name of data set to load')

-- training
cmd:option('-learningRate', 1e-3,'learning rate at t = 0')
cmd:option('-learningRateDecay', 0,'learning rate decay')
cmd:option('-momentum', 0.9,'momentum (SGD only)')
cmd:option('-weightDecay', 1e-3,'weight Decay')
cmd:option('-alpha', 0.95, 'alpha for optim.rmsprop')
cmd:option('-max_epochs', 1000, 'maximum epchos')
cmd:option('-savefreq', 100, 'epochs to save the model ')
cmd:text()

opt = cmd:parse(arg or {})


decoder_util = require 'decoder'
decoder = {}
if opt.type == 1 then
    decoder = decoder_util.create('../trainpic/codec_type1.txt', 8)
elseif opt.type == 2 or opt.type == 5 then
    decoder = decoder_util.create('../trainpic/codec_type2.txt', 5)
elseif opt.type == 3 or opt.type == 6 then
    decoder = decoder_util.create('../trainpic/chisayings.txt', 4)
elseif opt.type == 9 then
    decoder = decoder_util.create('../trainpic/codec_type9.txt', 4)
end
 -- print(decoder.label_size)
 -- print(decoder.ndigits)


print("loading data...")
data_util = require 'svhn_data'
trainset, validset = data_util.getFullset(opt.dataname)
print("trainset.size = ", trainset.size)
print("validset.size = ", validset.size)

if opt.gpuid > 0 then
    require 'cutorch'
    require 'cunn'
    print(cutorch.getDeviceCount(), "GPU devices detected")
    cutorch.setDevice(opt.gpuid)
    print("running on GPU", opt.gpuid)
    local freeMem, totalMem = cutorch.getMemoryUsage(opt.gpuid)
    print(string.format("GPU %d has %dM memory left, with %dM totally",
        opt.gpuid, freeMem/1000000, totalMem/1000000))
    trainset.data = trainset.data:cuda()
    trainset.labels = trainset.labels:cuda()
    validset.data = validset.data:cuda()
    validset.labels = validset.labels:cuda()
end

-- build a new model or use an existed model
model_util = require 'svhn_model'
model = nil
model_config = {
    picsize = {3, 50, 150},
    n_conv_layers = 3,  -- n_conv_layers == (#filter_num) - 1
    filter_num = {3, 4, 8, 16},
    filter_size = 5,
    dropout_value = opt.dropout,
    n_full_connect = 512,
    ndigits = 4,
    label_size = decoder.label_size,
}
if opt.model == '' then
    print("building CNN model...")
    if opt.type == 1 then
        model = model_util.createType1(opt.dropout)
    elseif opt.type == 2 then
        model = model_util.createType2(opt.dropout)
    elseif opt.type == 3 then
        model = model_util.createType3(decoder.label_size, opt.dropout)
    elseif opt.type == 5 then
        model = model_util.createType5()
    elseif opt.type == 6 then
        model = model_util.createType6()
    elseif opt.type == 9 then
        model = model_util.create(model_config)
    end
else
    print("loading CNN model...")
    model = torch.load(opt.model)
end

-- use GPU or CPU
if opt.gpuid > 0 then
    model = model:cuda()
end

x, dl_dx = model:getParameters()
print(string.format("%d parameters", x:size()[1]))

sgd_params = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
}

-- step training
step = function(trainset)
    local avg_loss = 0
    local avg_accuracy = 0
    local shuffle = torch.randperm(trainset.size)
    for i = 1, trainset.size do
        local input = trainset.data[shuffle[i]]
        local label = trainset.labels[shuffle[i]]
        if opt.gpuid > 0 then
            input = input:cuda()
            label = label:cuda()
        end

        local feval = function()
            local output = model:forward(input)
            local pL = output[1]:storage() -- output of length L
            local pS = output[2]:storage() -- output of character S[1..L]

            -- 1. calc loss
            local loss = pL[label[1] + 1]
            for j = 2, label[1]+1 do
                if label[j] ~= -1 then
                    local index = (j - 2) * decoder.label_size + label[j]
                    loss = loss + pS[index]
                end
            end

            -- 2. count correct labels
            prediction = decoder:output2label(output)
            if decoder:compareLabel(prediction, label) then
                avg_accuracy = avg_accuracy + 1
            end

            -- 3. calc outputGrad for pL and pS
            local l_grad = torch.Tensor(decoder.ndigits + 2):fill(0)
            local s_grad = torch.Tensor(decoder.ndigits * decoder.label_size):fill(0)
            if opt.gpuid > 0 then
                l_grad = l_grad:cuda()
                s_grad = s_grad:cuda()
            end
            l_grad[label[1] + 1] = -1
            for j = 2, label[1]+1 do
                local index = (j - 2) * decoder.label_size + label[j]
                s_grad[index] = -1
            end

            -- 4. backward the outputGrad
            dl_dx:zero()
            local outputGrad = {l_grad, s_grad:reshape(decoder.ndigits, decoder.label_size)}
            model:backward(input, outputGrad)

            return -loss, dl_dx
        end

        local _, loss = optim.sgd(feval, x, sgd_params)
        avg_loss = avg_loss + loss[1]
    end
    avg_loss = avg_loss / trainset.size
    avg_accuracy = avg_accuracy / trainset.size
    return avg_loss, avg_accuracy
end


validate = function(validset)
    -- local demo_per_size = validset.size / 10
    local avg_loss = 0
    local avg_accuracy = 0
    local shuffle = torch.randperm(validset.size)
    for i = 1, validset.size do
        local input = validset.data[shuffle[i]]
        local label = validset.labels[shuffle[i]]
        if opt.gpuid > 0 then
            input = input:cuda()
            label = label:cuda()
        end

        local output = model:forward(input)
        prediction = decoder:output2label(output)
        if decoder:compareLabel(prediction, label) then
            avg_accuracy = avg_accuracy + 1
        end

        -- calc validation loss
        local pL = output[1]:storage() -- output of length L
        local pS = output[2]:storage() -- output of character S[1..L]
        local loss = pL[label[1] + 1]
        for j = 2, label[1]+1 do
            if label[j] ~= -1 then
                local index = (j - 2) * decoder.label_size + label[j]
                loss = loss + pS[index]
            end
        end
        avg_loss = avg_loss + loss
    end
    avg_loss = - avg_loss / validset.size
    avg_accuracy = avg_accuracy / validset.size

    return avg_loss, avg_accuracy
end

train_loss_tensor = torch.Tensor(opt.max_epochs):fill(0)
valid_loss_tensor = torch.Tensor(opt.max_epochs):fill(0)
train_accuracy_tensor = torch.Tensor(opt.max_epochs):fill(0)
valid_accuracy_tensor = torch.Tensor(opt.max_epochs):fill(0)

local stoppingLR = opt.learningRate * 0.001
local stopwatch = 0
local last_v_loss = 100

for i = 1, opt.max_epochs do
    local timer = torch.Timer()

    -- training
    model:training()
    local train_loss, train_accuracy = step(trainset)
    train_loss_tensor[i] = train_loss
    train_accuracy_tensor[i] = train_accuracy

    -- validating
    model:evaluate()
    local valid_loss, valid_accuracy = validate(validset)
    valid_loss_tensor[i] = valid_loss
    valid_accuracy_tensor[i] = valid_accuracy

    print(string.format("epochs = %d,\tloss= %.3f, accu= %.3f, v_loss= %.3f, v_accu= %.3f, costs %.2fs",
        i, train_loss, train_accuracy, valid_loss, valid_accuracy, timer:time().real))

    -- cut the learning rate in half when valid_loss stops decreasing
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

    -- saving model
    if i % opt.savefreq == 0 then
        print("==================================")
        print(string.format("\nsaving model as %s\n", opt.savename))
        print("==================================")
        torch.save(opt.savename, model)
        print("saved, and learningRate is:", sgd_params.learningRate)
    end
end

-- print all information during the training process
for i = 1, opt.max_epochs do
    if train_loss_tensor[i] < 0.0000001 then
        break
    end
    print(string.format("i = %d,\tloss= %.3f, accu= %.3f, v_loss= %.3f, v_accu= %.3f",
        i, train_loss_tensor[i], train_accuracy_tensor[i],
        valid_loss_tensor[i], valid_accuracy_tensor[i]))
end
