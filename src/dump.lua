require 'image';
require 'io';
local decoder_util = require 'decoder'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type', 1, 'Which type of captcha to dump?')
cmd:option('-persize', 10, 'How many pictures to dump for every font?')
cmd:option('-datadir', '../synpic/type1/', 'Where to find pictures to dump?')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
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

local data, labels
local decoder = {}

if opt.type == 1 then
    data = torch.Tensor(fullset.size, 3, 50, 200)
    labels = torch.IntTensor(fullset.size, 9):fill(0)
    decoder = decoder_util.create('../synpic/codec_type1.txt', 8)
elseif opt.type == 2 then
    data = torch.Tensor(fullset.size, 3, 53, 160)
    labels = torch.IntTensor(fullset.size, 6):fill(0)
    decoder = decoder_util.create('../synpic/codec_type2.txt', 5)
elseif opt.type == 3 then
    data = torch.Tensor(fullset.size, 3, 53, 160)
    labels = torch.IntTensor(fullset.size, 5):fill(0)
end
-- print(string.format("size of data:\n%s", #data))
-- print(string.format("size of labels:\n%s", #labels))
for i = 1, opt.persize do
    if i % (opt.persize / 10) == 0 then
        print(i / opt.persize * 100 .. "% finished") -- progress bar
    end
    for j = 1, #fonts do
        local img = image.load(path.join(opt.datadir, fonts[j] .. i-1 .. '.jpg'))
        data[(i-1) * #fonts + j] = img
        local file = io.open(path.join(opt.datadir, fonts[j] .. i-1 .. '.gt.txt'), 'r')
        local str = file:read()
        file:close()
        labels[(i-1) * #fonts + j] = decoder:str2label(str)
    end
end

fullset.data = data
fullset.labels = labels

print("\nsaving", opt.savename)
torch.save(opt.savename, fullset)
