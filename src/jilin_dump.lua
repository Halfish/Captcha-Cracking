require 'image'
require 'io'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-totalsize', 10000, 'how many pictures to dump for every type')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:option('-shuffle', false, 'shuffle or not')
cmd:option('-verbose', false, 'whether or not to print filenames')
cmd:text()
local opt = cmd:parse(arg or {})

local decoder_util = require 'decoder'
local decoder = decoder_util.create('../trainpic/codec_type8.txt', 7)

local fullset = {}
fullset.size = opt.totalsize
print('fullset size = ', fullset.size)
local data = torch.Tensor(fullset.size, 3, 50, 300)
local label = torch.IntTensor(fullset.size, 8):fill(0)

local paths = require 'paths'
local count = 0
local rootdir = '../trainpic/type8-real/'
for file in paths.iterfiles(rootdir) do
    count = count + 1
    if count > opt.totalsize then break end
    data[count] = image.load(rootdir .. file)
    local str = string.split(file, '_')[2]
    label[count] = decoder:str2label(str)
end

if opt.shuffle then
    print('in shuffle ..')
    fullset.data = data:clone():fill(0)
    fullset.label = label:clone():fill(0)
    local shuffle = torch.randperm(fullset.size)
    for i = 1, fullset.size do
        fullset.data[i] = data[shuffle[i]]
        fullset.label[i] = label[shuffle[i]]
    end
else
    fullset.data = data
    fullset.label = label
end

print('\nsaving', opt.savename)
torch.save(opt.savename, fullset)
