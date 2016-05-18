local cmd = torch.CmdLine()
cmd:text()
cmd:option('-size', 100, 'how many pictures to dump')
cmd:option('-picdir', '../trainpic/nacao/', 'directory of pictures to train')
cmd:option('-type', 3, 'which type to dump, default is type3')
cmd:text()
local opt = cmd:parse(arg or {})

require 'image'
local decoder_util = require 'decoder'

local fullset = {}
fullset.size = opt.size

local data, label, decoder
if opt.type == 3 then
    data = torch.Tensor(fullset.size, 3, 50, 140)
    label = torch.IntTensor(fullset.size, 7)
    decoder = decoder_util.create('../trainpic/codec_nacao.txt', 6)
end

function loaddata(index, filename, l)
end

local filename = path.join(opt.picdir, 'type3.txt')
local file = io.open(filename, 'r')
for i = 1, opt.size do
    local picname = '../trainpic/nacao/type3/'.. i .. '.jpg'
    data[i] = image.load(picname, 3)
    label[i] = decoder:str2label(file:read())
end

fullset.data = data
fullset.label = label
torch.save('fullset.dat', fullset)
