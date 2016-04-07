require 'decoder'
require 'image'

local data_util = {}

function data_util.getFullset(name, split_rate)
    local fullname = './' .. name
    split_rate = split_rate or 0.9
    fullset = torch.load(fullname)

    -- from RGB to YUV
    for i = 1, fullset.size do
        fullset.data[i] = image.rgb2yuv(fullset.data[i])
    end

    -- normalization
    local channels = {'y', 'u', 'v'}
    local mean = {}
    local std = {}
    for i, channel in ipairs(channels) do
        mean[i] = fullset.data[{{}, i, {}, {}}]:mean()
        std[i] = fullset.data[{{}, i, {}, {}}]:std()
        fullset.data[{{}, i, {}, {}}]:add(-mean[i])
        fullset.data[{{}, i, {}, {}}]:div(std[i])
    end

    -- split data into train and validation
    local trainset = {}
    local validset = {}
    trainset.size = fullset.size * split_rate
    validset.size = fullset.size - trainset.size
    trainset.data = fullset.data[{{1, trainset.size}}]
    validset.data = fullset.data[{{1 + trainset.size, fullset.size}}]
    trainset.label = fullset.label[{{1, trainset.size}}]
    validset.label = fullset.label[{{1 + trainset.size, fullset.size}}]

    return trainset, validset
end

return data_util
