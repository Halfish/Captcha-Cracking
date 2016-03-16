require 'nn'
require 'nngraph'

local model_util = {}

-- create a model
function model_util.create(dropout_value, filter_size)
    filter_size = filter_size or 5
    -- stage 1: build an input for nngraph
    local input = nn.Identity()()
    local L1 = nn.Reshape(3, 50, 200)(input)
    local L2 = nn.MulConstant(1.0 * 3.2)(L1)

    -- stage 2: convolution -> squashing --> L2 pooling
    local C1 = nn.SpatialConvolutionMM(3, 4, filter_size, filter_size)(L2)
    local R1 = nn.ReLU()(C1) -- using Rectified Linear Unit as transfer function
    local P1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R1)
    -- local kernel1 = torch.ones(11)
    -- local N1 = nn.SpatialSubtractiveNormalization(64, kernel1)(P1)

    -- stage 3: convolution -> squashing --> L2 pooling
    local C2 = nn.SpatialConvolutionMM(4, 8, filter_size, filter_size)(P1)
    local R2 = nn.ReLU()(C2)
    local P2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(R2)
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

return model_util
