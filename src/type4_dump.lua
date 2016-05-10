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
    codec_mapper[codec[i]] = i
end

-- define arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:option('-province', 'nx', 'which province to choice? [chq, gs, nx, tj, jx, small, nacao]')
cmd:option('-typename', 'num', 'num or symb')
cmd:option('-picdir', '../trainpic/type4/', 'directory of pictures to train')
cmd:option('-size', 99, 'how many pictures to dump')
cmd:text()
local opt = cmd:parse(arg or {})


-- define regular expression
local picname = opt.province .. '_%d+_[13]_%d.png'
if opt.typename == 'symb' then
    picname = opt.province .. '_%d+_[2]_[%+-%*].png'
end
local dumpname = 'type4_' .. opt.province .. '_' .. opt.typename .. '.dat'

-- define data and label
local label = torch.IntTensor(opt.size)
local data 
if opt.province == 'gs' then
    data = torch.Tensor(opt.size, 3, 18, 18)
elseif opt.province == 'jx' then
    if opt.typename == 'num' then
        data = torch.Tensor(opt.size, 1, 30, 25)
    elseif opt.typename == 'symb' then
        data = torch.Tensor(opt.size, 1, 35, 35)
    end
elseif opt.province == 'small' then
    if opt.typename == 'num' then
        data = torch.Tensor(opt.size, 1, 15, 11)
    elseif opt.typename == 'symb' then
        data = torch.Tensor(opt.size, 1, 16, 16)
    end
elseif opt.province == 'chq' or opt.province == 'nx' or opt.province == 'tj' then
    data = torch.Tensor(opt.size, 1, 30, 30)
elseif opt.province == 'nacao' then
    data = torch.Tensor(opt.size, 3, 30, 20)
end

-- find every file which matches the regular expression
local i = 1
for file in paths.iterfiles(opt.picdir) do
    if string.find(file, picname) then
        data[i] = image.load(path.join(opt.picdir, file))
        local l = string.split(file, '%.')[1]
        l = string.split(l, '_')[4]
        if opt.typename == 'num' then
            label[i] = tonumber(l) + 1
        elseif opt.typename == 'symb' then
            label[i] = codec_mapper[l]
        end
        if i == opt.size then
            break
        else
            i = i + 1
        end
    end
end

if i ~= opt.size then
    print(string.format('i = %d, size = %d', i, opt.size))
    print('inconsistent size of data, may not efficient pictures')
end

-- dump data
fullset.size = label:size()[1]
fullset.label = label
fullset.data = data
print('saving data as ' .. dumpname)
torch.save(dumpname, fullset)
