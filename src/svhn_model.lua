require 'nn'
require 'nngraph'

local model_util = {}

function model_util.create(model_config)
    local picsize = model_config.picsize                -- like {3, 50, 150}
    local filter_num = model_config.filter_num          -- like {3, 4, 8, 16}, filter num 
    local filter_size = model_config.filter_size        -- like 5, filter size for conv layer 
    local dropout_value = model_config.dropout_value    -- like 0.5, dropout 
    local n_full_connect = model_config.n_full_connect  -- like 128, fully connected layer num
    local ndigits = model_config.ndigits                -- like 4, maximum number of digits
    local label_size = model_config.label_size          -- like 20, types of labels

    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.View(picsize[1], picsize[2], picsize[3])(input)

    local L_conv = L1
    local size1 = picsize[2]
    local size2 = picsize[3] 
    -- stage 2: convolution layers
    for i = 1, #filter_num - 1 do
        local L_C = nn.SpatialConvolutionMM(filter_num[i], filter_num[i+1], 
        filter_size, filter_size)(L_conv)
        local L_R = nn.ReLU()(L_C) -- using Rectified Linear Unit as transfer function
        local L_P = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(L_R)
        L_conv = L_P
        size1 = math.floor((size1 - filter_size + 1) / 2)
        size2 = math.floor((size2 - filter_size + 1) / 2)
    end

    -- stage 3: dropout
    local L_D = nn.Dropout(dropout_value)(L_conv)    -- dropout

    -- stage 4: fully connected layer
    local size = size1 * size2 * filter_num[#filter_num]
    local L2 = nn.Reshape(size)(L_D)
    local L3 = nn.Linear(size, n_full_connect)(L2)
    local R4 = nn.ReLU()(L3)

    -- stage 5A: put feature R4 into LogSoftMax of L
    local L4 = nn.Linear(n_full_connect, ndigits + 2)(R4)
    local R5 = nn.ReLU()(L4)
    local L = nn.LogSoftMax()(R5)

    -- stage 5B: put feature R4 into LogSoftMax of S
    local L5 = nn.Linear(n_full_connect, ndigits * label_size)(R4)
    local R6 = nn.ReLU()(L5)
    local S = nn.LogSoftMax()(nn.View(ndigits, label_size)(R6))

    -- stage 6: build model
    local model = nn.gModule({input}, {L, S})

    -- initialize model with xavier
    model = require('weight-init')(model, 'xavier')

    return model
end

function model_util.createType1()
    model_config = {
        picsize = {3, 50, 200},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 8,
        label_size = 20
    }
    return model_util.create(model_config)
end

function model_util.createType2()
    model_config = {
        picsize = {3, 53, 160},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 5,
        label_size = 15
    }
    return model_util.create(model_config)
end

function model_util.createType3()
    model_config = {
        picsize = {3, 53, 160},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 2084,
        ndigits = 4,
        label_size = 3769
    }
    return model_util.create(model_config)
end

function model_util.createType5()
    model_config = {
        picsize = {3, 40, 180},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 5,
        label_size = 15
    }
    return model_util.create(model_config)
end

function model_util.createType6()
    model_config = {
        picsize = {3, 40, 180},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 4,
        label_size = 294
    }
    return model_util.create(model_config)
end

function model_util.createType7()
    model_config = {
        picsize = {3, 40, 260},
        filter_num = {3, 8, 16, 32},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 7,
        label_size = 45
    }
    return model_util.create(model_config)
end

function model_util.createType8()
    model_config = {
        picsize = {3, 50, 300},
        filter_num = {3, 8, 16, 32},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 7,
        label_size = 42
    }
    return model_util.create(model_config)
end


function model_util.createType9()
    model_config = {
        picsize = {3, 50, 150},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = opt.dropout,
        n_full_connect = 512,
        ndigits = 4,
        label_size = 3769
    }
    return model_util.create(model_config)
end

function model_util.createNacao3()
    model_config = {
        picsize = {1, 50, 140},
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = opt.dropout,
        n_full_connect = 128,
        ndigits = 6,
        label_size = 27
    }
    return model_util.create(model_config)
end

return model_util
