--[[

This Class 'decoder_util' offers some basic functions to change forms of label from one to another

Bacally, there are three froms(if we assume self.label_size is 8):

    1. str: string format, which is also readable
        str = '1加上3等于？'

    2. label: real label, Tensor format
        label = {label_length, 1st, 2nd,..., 8th}

    3. output: output Tensor for Torch, including two inner Tensors
        output = {
            pL: 10 Tensor, probability of length, 0,1,...,8,8+
            pS: 20x8 Tensor, probability for 8 char selection
        }

    8 is the self.ndigits
    20 is self.label_size, 20-classification problem for the model 

]]--

require 'io'
local path = require 'pl.path'

local decoder_util = {}     -- declare class decoder_util
decoder_util.__index = decoder_util     -- just syntactic sugar

function decoder_util.create(codec_dir, ndigits)
    -- constructor for Class decoder_util
    local self = {}
    setmetatable(self, decoder_util)

    self.mapper, self.rev_mapper = decoder_util.get_mapper(codec_dir)
    -- self.label_size = #self.mapper -- this doesn't work
    self.label_size = 0
    for k, v in pairs(self.mapper) do
        self.label_size = self.label_size + 1
    end
    self.ndigits = ndigits

    return self
end

-- STATIC method, inspired by zhangzibin@github
-- get table with vary length from str, which may include chinese unicode character
function decoder_util.str2vocab(str)
    local vocab = {}
    local len  = #str
    local left = 0
    local arr  = {0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc}
    local start = 1
    local wordLen = 0
    while len ~= left do
        local tmp = string.byte(str, start)
        local i   = #arr
        while arr[i] do
            if tmp >= arr[i] then
                break
            end
            i = i - 1
        end
        wordLen = i + wordLen
        local tmpString = string.sub(str, start, wordLen)
        start = start + i
        left = left + i
        vocab[#vocab+1] = tmpString
    end
    return vocab
end

-- STATIC method, get chinese vocabulary mapper and rev_mapper from file,
-- you can just print mapper, and rev_mapper to see what does it looks like
function decoder_util.get_mapper(filename)
    local file = io.open(filename, 'r')
    local str = file:read()
    local vocab = decoder_util.str2vocab(str)
    local mapper = {}
    local rev_mapper = {}
    for i = 1, #vocab do
        mapper[vocab[i]] = true
    end
    local count = 1
    for w, _ in pairs(mapper) do
        mapper[w] = count
        table.insert(rev_mapper, w)
        count = count + 1
    end
    return mapper, rev_mapper
end

function decoder_util:str2label(str)
    local label = decoder_util.str2vocab(str)
    for i = 1, #label do
        label[i] = self.mapper[label[i]]
    end
    local tensorLabel = torch.Tensor(self.ndigits + 1):fill(0)
    tensorLabel[1] = #label
    for i = 1, #label do
        tensorLabel[i+1] = label[i]
    end
    return tensorLabel -- label is a table
end

function decoder_util:label2str(label)
    local str = ''
    for i = 1, label[1] do -- so label should be a table
        str = str .. self.rev_mapper[label[i+1]]
    end
    return str
end

-- parse network's output into standard label
function decoder_util:output2label(output)
    local prob_L = output[1]
    local prob_S, index = output[2]:max(2)
    prob_S = prob_S:reshape(prob_S:size(1))
    index = index:reshape(index:size(1))
    local max_prob = prob_L[1] -- max_prob = the prob of length being 0
    local length = 0 -- default length is zero
    local cul = 0
    for i = 1, self.ndigits do
        cul = prob_S[i] + cul
        if (cul + prob_L[i+1]) > max_prob then
            max_prob = cul + prob_L[i+1] 
            length = i
        end
    end

    local label = torch.Tensor(self.ndigits + 1):fill(0)
    label[1] = length
    for i = 1, length do
        label[i+1] = index[i]
    end
    return label
end

-- only if these two labels are the same will we return true
function decoder_util:compareLabel(label1, label2)
    -- compare length first
    if label1[1] ~= label2[1] then
        return false
    end
    for i = 1, label1[1] do
        if label1[i+1] ~= label2[i+1] then
            return false
        end
    end
    return true
end

-- using regular expression to change simple '1+1' into '1加上1等于？'
function decoder_util:simple2str_type1(simple)
    str = simple
    str = string.gsub(str, '+', '加上')
    str = string.gsub(str, '-', '减去')
    str = string.gsub(str, '*', '乘以')
    str = string.gsub(str, '/', '除以')
    return str .. '等于？'
end

-- using regular expression to change simple '1+1' into '壹加壹等于'
lang_map = {'零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'}
function decoder_util:simple2str_type2(simple)
    str = simple
    str = string.gsub(str, '+', '加')
    str = string.gsub(str, '-', '减')
    str = string.gsub(str, '*', '乘')
    for i = 1, 10 do
        str = string.gsub(str, (i-1), lang_map[i])
    end
    return str .. '等于'
end

return decoder_util
