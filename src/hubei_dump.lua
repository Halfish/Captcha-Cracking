require 'image'
require 'io'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-persize', 100, 'how many pictures to dump for every type')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:option('-verbose', false, 'whether or not to print filenames')
cmd:text()
local opt = cmd:parse(arg or {})

local decoder_util = require 'decoder'
local decoder = decoder_util.create('../trainpic/codec_type10.txt', 1)

local fullset = {}
fullset.size = opt.persize * decoder.label_size
print('fullset size = ', fullset.size)
local data = torch.Tensor(fullset.size, 1, 40, 22)
local label = torch.IntTensor(fullset.size):fill(0)

local paths = require 'paths'
local count = 0
local rootdir = '../trainpic/type10/'
for dir in paths.iterdirs(rootdir) do
    if opt.verbose then xlua.progress(count, fullset.size) end
    local stopcount = 0
    for file in paths.iterfiles(rootdir .. dir) do
        stopcount = stopcount + 1
        if stopcount > opt.persize then break end
        local filename = path.join(rootdir .. dir, file)
        local img = image.load(filename, 1)
        count = count + 1
        data[count][1] = img
        if not decoder.mapper[dir] then print('nil' .. dir .. tostring(decoder.mapper['阿'])) end
        label[count] = decoder.mapper[dir]
    end
end

fullset.data = data:clone():fill(0)
fullset.label = label:clone():fill(0)
local shuffle = torch.randperm(fullset.size)
for i = 1, fullset.size do
    fullset.data[i] = data[shuffle[i]]
    fullset.label[i] = label[shuffle[i]]
end

print('\nsaving', opt.savename)
torch.save(opt.savename, fullset)
