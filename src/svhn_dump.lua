require 'image';
require 'io';
local decoder_util = require 'decoder'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type', 1, 'Which type of captcha to dump?')
cmd:option('-persize', 10, 'How many pictures to dump for every font?')
cmd:option('-datadir', '../trainpic/type1/', 'Where to find pictures to dump?')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:option('-format', '.jpg', 'jpg or png?')
cmd:text()
local opt = cmd:parse(arg or {})

local fonts = {}
for file in paths.iterfiles('../fonts/') do
    fonts[#fonts + 1] = string.split(file, '%.')[1]
end
print("Fonts:")
print(fonts)
print('')

local fullset = {}
fullset.size = opt.persize * #fonts

local data, label
local decoder = {}

if opt.type == 1 then
    data = torch.Tensor(fullset.size, 3, 50, 200)
    label = torch.IntTensor(fullset.size, 9):fill(0)
    decoder = decoder_util.create('../trainpic/codec_type1.txt', 8)
elseif opt.type == 2 then
    data = torch.Tensor(fullset.size, 3, 53, 160)
    label = torch.IntTensor(fullset.size, 6):fill(0)
    decoder = decoder_util.create('../trainpic/codec_type2.txt', 5)
elseif opt.type == 3 then
    data = torch.Tensor(fullset.size, 3, 53, 160)
    label = torch.IntTensor(fullset.size, 5):fill(0)
    decoder = decoder_util.create('../trainpic/chisayings.txt', 4)
elseif opt.type == 5 then
    data = torch.Tensor(fullset.size, 3, 40, 180)
    label = torch.IntTensor(fullset.size, 6):fill(0)
    decoder = decoder_util.create('../trainpic/codec_type2.txt', 5)
elseif opt.type == 6 then
    data = torch.Tensor(fullset.size, 3, 40, 180)
    label = torch.IntTensor(fullset.size, 5):fill(0)
    decoder = decoder_util.create('../trainpic/chisayings.txt', 4)
elseif opt.type == 9 then
    data = torch.Tensor(fullset.size, 3, 50, 150)
    label = torch.IntTensor(fullset.size, 5):fill(0)
    decoder = decoder_util.create('../trainpic/codec_type9.txt', 4)
elseif opt.type == 10 then
    data = torch.Tensor(fullset.size, 3, 40, 22)
    label = torch.IntTensor(fullset.size):fill(0)
    decoder = decoder_util.create('../trainpic/chisayings.txt', 1)
end
-- print(string.format("size of data:\n%s", #data))
-- print(string.format("size of label:\n%s", #label))
for i = 1, opt.persize do
    if i % (opt.persize / 10) == 0 then
        print(i / opt.persize * 100 .. "% finished") -- progress bar
    end
    for j = 1, #fonts do
        local img = image.load(path.join(opt.datadir, fonts[j] .. i-1 .. opt.format), 3)
        if opt.type == 9 then
            -- type 9 needs preprocess
            k = image.gaussian(3)
            img = image.convolve(img, k, 'same')
            img[img:lt(0.5)] = 0
            img[img:ge(0.5)] = 1
        end
        data[(i-1) * #fonts + j] = img
        local file = io.open(path.join(opt.datadir, fonts[j] .. i-1 .. '.gt.txt'), 'r')
        local str = file:read()
        file:close()
        if opt.type == 10 then
            label[(i-1) * #fonts + j] = decoder.mapper[str]
        else
            label[(i-1) * #fonts + j] = decoder:str2label(str)
        end
    end
end

fullset.data = data
fullset.label = label

print("\nsaving", opt.savename)
torch.save(opt.savename, fullset)
