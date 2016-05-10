local cmd = torch.CmdLine()
cmd:text()
cmd:option('-size', 9, 'how many pictures to dump')
cmd:option('-picdir', '../trainpic/type9/', 'directory of pictures to train')
cmd:text()
local opt = cmd:parse(arg or {})

require 'image'

local fullset = {}
fullset.size = 4 * opt.size

local data = torch.Tensor(fullset.size, 3, 41, 34)
local label = torch.IntTensor(fullset.size)

local decoder_util = require 'decoder'
local decoder = decoder_util.create('../trainpic/codec_type9.txt')

function cut(index, filename, l)
    local img = image.load(filename)
    local alphabeta = {}
    alphabeta[1] = img[{{}, {8, 48}, {28, 61}}]
    alphabeta[2] = img[{{}, {8, 48}, {58, 91}}]
    alphabeta[3] = img[{{}, {8, 48}, {90, 123}}]
    alphabeta[4] = img[{{}, {8, 48}, {117, 150}}]
    for i = 1, 4 do
        data[index * 4 + i] = alphabeta[i]
        label[index * 4 + i] = decoder.mapper[string.sub(l, i, i)]
    end
end

local filename = path.join(opt.picdir, 'label.txt')
local file = io.open(filename, 'r')
for i = 1, opt.size do
    local l = file:read()
    cut(i - 1, '../trainpic/type9/'.. i .. '.jpg', l)
end

fullset.data = data
fullset.label = label
torch.save('fullset.dat', fullset)

