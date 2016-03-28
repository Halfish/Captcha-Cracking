--[[
--  Usage: Dump pictures from labeled data for type4
--  for example:
--      th type4_dump -province nx -typename num -picdir ../trainpic/type4 
--]]--
require 'image'

-- define codec
local fullset = {}
local codec = {'+', '-', '*'}
local codec_mapper = {}
for i = 1, #codec do
    codec_mapper[codec[i]] = i - 1
end

-- define arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:option('-province', 'nx', 'which province to choice? [chq, gs, nx, tj, jx]')
cmd:option('-typename', 'num', 'num or symb')
cmd:option('-picdir', '../trainpic/type4/', 'directory of pictures to train')
cmd:text()
local opt = cmd:parse(arg or {})

local size = 99
if opt.typename == 'num' then
    size = 198
end

-- define regular expression
local picname = opt.province .. '_%d+_[13]_%d.png'
if opt.typename == 'symb' then
    picname = opt.province .. '_%d+_[2]_[%+-%*].png'
end
local dumpname = 'type4_' .. opt.province .. '_' .. opt.typename .. '.dat'

-- define data and label
local label = torch.IntTensor(size)
local data 
if opt.province == 'gs' then
    data = torch.Tensor(size, 3, 18, 18)
elseif opt.province == 'jx' then
    if opt.typename == 'num' then
        data = torch.Tensor(size, 1, 30, 25)
    elseif opt.typename == 'symb' then
        data = torch.Tensor(size, 1, 35, 35)
    end
elseif opt.province == 'chq' or opt.province == 'nx' or opt.province == 'tj' then
    data = torch.Tensor(size, 1, 30, 30)
end

-- find every file which matches the regular expression
local i = 1
for file in paths.iterfiles(opt.picdir) do
    if string.find(file, picname) then
        data[i] = image.load(path.join(opt.picdir, file))
        local l = string.split(file, '%.')[1]
        l = string.split(l, '_')[4]
        if opt.typename == 'num' then
            label[i] = tonumber(l)
        elseif opt.typename == 'symb' then
            label[i] = codec_mapper[l]
        end
        if i == num then
            break
        else
            i = i + 1
        end
    end
end

-- dump data
fullset.size = label:size()[1]
fullset.label = label
fullset.data = data
print('saving data as ' .. dumpname)
torch.save(dumpname, fullset)
