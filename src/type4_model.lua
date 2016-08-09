require 'nn'

local model_util = {}

function model_util.create(model_config)
    local picsize = model_config.picsize                -- like {3, 50, 150}
    local filter_num = model_config.filter_num          -- like {3, 4, 8, 16}, filter num 
    local filter_size = model_config.filter_size        -- like 5, filter size for conv layer 
    local dropout_value = model_config.dropout_value    -- like 0.5, dropout 
    local n_full_connect = model_config.n_full_connect  -- like 128, fully connected layer num
    local nclass = model_config.nclass

    -- stage 1: build an input for nngraph
    local model = nn.Sequential()
    model:add(nn.View(picsize[1], picsize[2], picsize[3]))

    -- stage 2: convolution layers
    local size1, size2 = picsize[2], picsize[3]
    local dW, dH, padW, padH = 1, 1, (filter_size - 1) / 2, (filter_size - 1) / 2
    for i = 1, #filter_num - 1 do
        model:add(nn.SpatialConvolutionMM(filter_num[i], filter_num[i+1], filter_size, filter_size, dW, dH, padW, padH))
        model:add(nn.SpatialBatchNormalization(filter_num[i+1]))
        model:add(nn.ReLU()) -- using Rectified Linear Unit as transfer function
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
        size1 = math.floor(size1 / 2)
        size2 = math.floor(size2 / 2)
    end

    -- stage 3: dropout
    model:add(nn.Dropout(dropout_value))    -- dropout

    -- stage 4: fully connected layer
    local size = size1 * size2 * filter_num[#filter_num]
    model:add(nn.View(size))
    model:add(nn.Linear(size, n_full_connect[1]))
    model:add(nn.ReLU())
    model:add(nn.Linear(n_full_connect[1], n_full_connect[2]))
    model:add(nn.ReLU())

    -- stage 5: softmax classifier
    model:add(nn.Linear(n_full_connect[2], nclass))
    model:add(nn.LogSoftMax())

    model = require('weight-init')(model, 'xavier')

    return model
end


function model_util.createType10()
    model_config = {
        picsize = {1, 22, 40},
        filter_num = {1, 64, 128},
        filter_size = 3,
        dropout_value = 0.5,
        n_full_connect = {256, 256},
        nclass = 1454
    }
    return model_util.create(model_config)
end

return model_util
