require 'nn'
require 'nngraph'

local model_util = {}

function model_util.create(model_config)
    local picsize = model_config.picsize                -- like {3, 50, 150}
    local n_conv_layers = model_config.n_conv_layers    -- like 3 or 4, conv layers
    local filter_num = model_config.filter_num          -- like {3, 4, 8, 16}, filter num 
    local filter_size = model_config.filter_size        -- like 5, filter size for conv layer 
    local dropout_value = model_config.dropout_value    -- like 0.5, dropout 
    local n_full_connect = model_config.n_full_connect  -- like 128, fully connected layer num
    local ndigits = model_config.ndigits              -- like 4, maximum number of digits
    local label_size = model_config.label_size          -- like 20, types of labels

    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.View(picsize[1], picsize[2], picsize[3])(input)

    local L_conv = L1
    local size1 = picsize[2]
    local size2 = picsize[3] 
    -- stage 2: convolution layers
    for i = 1, n_conv_layers do
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


-- create a model for type 1 CAPTCHA
function model_util.createType1(dropout_value, filter_size)
    filter_size = filter_size or 5
    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.Reshape(3, 50, 200)(input)
    local L2 = nn.MulConstant(1.0 * 3.2)(L1)

    -- stage 2: convolution -> squashing --> L2 pooling
    local C1 = nn.SpatialConvolutionMM(3, 4, filter_size, filter_size)(L2)
    local R1 = nn.ReLU()(C1) -- using Rectified Linear Unit as transfer function
    local P1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R1)
    -- This should be added up, but Torch haven't developed a GPU ver yet.
    -- local kernel1 = torch.ones(11)
    -- local N1 = nn.SpatialSubtractiveNormalization(64, kernel1)(P1)

    -- stage 3: convolution -> squashing --> L2 pooling
    local C2 = nn.SpatialConvolutionMM(4, 8, filter_size, filter_size)(P1)
    local R2 = nn.ReLU()(C2)
    local P2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R2)
    -- This should be added up, but Torch haven't developed a GPU ver yet.
    -- local kernel2 = torch.ones(7)
    -- local N2 = nn.SpatialSubtractiveNormalization(128, kernel2)(P2)

    local C3 = nn.SpatialConvolutionMM(8, 16, filter_size, filter_size)(P2)
    local R3 = nn.ReLU()(C3)
    local P3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R3)
    local D1 = nn.Dropout(dropout_value)(P3)

    -- stage 4: fully connected layer
    local L3 = nn.Reshape(2 * 21 * 16)(D1)
    local L4 = nn.Linear(2 * 21 * 16, 128)(L3)
    local R3 = nn.ReLU()(L4)

    -- stage 5: put feature R3 into LogSoftMax of L and S
    local L = nn.LogSoftMax()(nn.ReLU()(nn.Linear(128, 10)(R3)))
    local S = nn.LogSoftMax()(nn.View(8, 20)(nn.ReLU()(nn.Linear(128, 160)(R3))))

    -- build model
    local model = nn.gModule({input}, {L, S})

    -- init model with xavier
    model = require('weight-init')(model, 'xavier')

    return model
end

-- create a model for type 2 CAPTCHA
function model_util.createType2(dropout_value, filter_size)
    filter_size = filter_size or 5
    local ndigits = 5
    local label_size = 15
    local picsize = {3, 53, 160}

    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.View(picsize[1], picsize[2], picsize[3])(input)

    -- stage 2: convolution -> squashing --> L2 pooling
    local C1 = nn.SpatialConvolutionMM(picsize[1], 4, filter_size, filter_size)(L1)
    local R1 = nn.ReLU()(C1) -- using Rectified Linear Unit as transfer function
    local P1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R1)

    -- stage 3: convolution -> squashing --> L2 pooling
    local C2 = nn.SpatialConvolutionMM(4, 8, filter_size, filter_size)(P1)
    local R2 = nn.ReLU()(C2)
    local P2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R2)

    -- stage 4: convolution -> squashing --> L2 pooling
    local C3 = nn.SpatialConvolutionMM(8, 16, filter_size, filter_size)(P2)
    local R3 = nn.ReLU()(C3)
    local P3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R3)
    local D1 = nn.Dropout(dropout_value)(P3)    -- dropout

    -- stage 4: fully connected layer
    local size1 = math.floor(math.floor((math.floor(picsize[2] - 4) / 2 - 4) / 2 - 4) / 2)
    local size2 = math.floor(math.floor((math.floor(picsize[3] - 4) / 2 - 4) / 2 - 4) / 2)
    local L2 = nn.Reshape(size1 * size2 * 16)(D1)
    local L3 = nn.Linear(size1 * size2 * 16, 128)(L2)
    local R4 = nn.ReLU()(L3)

    -- stage 5: put feature R4 into LogSoftMax of L and S
    local L4 = nn.Linear(128, ndigits + 2)(R4)
    local R5 = nn.ReLU()(L4)
    local L = nn.LogSoftMax()(R5)

    local L5 = nn.Linear(128, ndigits * label_size)(R4)
    local R6 = nn.ReLU()(L5)
    local S = nn.LogSoftMax()(nn.View(ndigits, label_size)(R6))

    -- build model
    local model = nn.gModule({input}, {L, S})

    -- init model with xavier
    model = require('weight-init')(model, 'xavier')

    return model
end

function model_util.createType3(label_size, dropout_value, filter_size)
    filter_size = filter_size or 5
    local ndigits = 4
    local kernel_num = 4
    local kernel_full =  2048
    -- local label_size = 3769
    local picsize = {3, 53, 160}

    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.View(picsize[1], picsize[2], picsize[3])(input)

    -- stage 2: convolution -> squashing --> L2 pooling
    local C1 = nn.SpatialConvolutionMM(picsize[1], kernel_num, filter_size, filter_size)(L1)
    local R1 = nn.ReLU()(C1) -- using Rectified Linear Unit as transfer function
    local P1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R1)

    -- stage 3: convolution -> squashing --> L2 pooling
    local C2 = nn.SpatialConvolutionMM(kernel_num, kernel_num * 2, filter_size, filter_size)(P1)
    local R2 = nn.ReLU()(C2)
    local P2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R2)

    -- stage 4: convolution -> squashing --> L2 pooling
    local C3 = nn.SpatialConvolutionMM(kernel_num * 2, kernel_num * 4, filter_size, filter_size)(P2)
    local R3 = nn.ReLU()(C3)
    local P3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R3)
    local D1 = nn.Dropout(dropout_value)(P3)    -- dropout

    -- stage 4: fully connected layer
    local size1 = math.floor(math.floor((math.floor(picsize[2] - 4) / 2 - 4) / 2 - 4) / 2)
    local size2 = math.floor(math.floor((math.floor(picsize[3] - 4) / 2 - 4) / 2 - 4) / 2)
    local L2 = nn.Reshape(size1 * size2 * kernel_num * 4)(D1)
    local L3 = nn.Linear(size1 * size2 * kernel_num * 4, kernel_full)(L2)
    local R4 = nn.ReLU()(L3)

    -- stage 5: put feature R4 into LogSoftMax of L and S
    local L4 = nn.Linear(kernel_full, ndigits + 2)(R4)
    local R5 = nn.ReLU()(L4)
    local L = nn.LogSoftMax()(R5)

    local L5 = nn.Linear(kernel_full, ndigits * label_size)(R4)
    local R6 = nn.ReLU()(L5)
    local S = nn.LogSoftMax()(nn.View(ndigits, label_size)(R6))

    -- build model
    local model = nn.gModule({input}, {L, S})

    -- init model with xavier
    model = require('weight-init')(model, 'xavier')

    return model
end


function model_util.createType5()
    model_config = {
        picsize = {3, 40, 180},
        n_conv_layers = 3,
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
        n_conv_layers = 3,
        filter_num = {3, 4, 8, 16},
        filter_size = 5,
        dropout_value = 0.5,
        n_full_connect = 128,
        ndigits = 4,
        label_size = 294
    }
    return model_util.create(model_config)
end


return model_util
