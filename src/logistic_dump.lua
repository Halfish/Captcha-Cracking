require 'image'
require 'io'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type', 23, 'Which type to dump, 23 or 56')
cmd:option('-persize', 10, 'how many pictures to dump for every type')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:option('-verbose', false, 'whether or not to print filenames')
cmd:text()
local opt = cmd:parse(arg or {})

local dirs = {}
local fullset = {}
local data, label
fullset.size = opt.persize * 2
label = torch.IntTensor(fullset.size):fill(0)
local format = ''

if opt.type == 23 then
    dirs = {'../trainpic/logistic/type2', '../trainpic/logistic/type3'}
    data = torch.Tensor(fullset.size, 3, 53, 160)
    format = '.jpg'
elseif opt.type == 56 then
    dirs = {'../trainpic/logistic/type5', '../trainpic/logistic/type6'}
    data = torch.Tensor(fullset.size, 3, 40, 180)
    format = '.png'
end



for i = 1, opt.persize do
    for j = 1, 2 do
        local filename = path.join(dirs[j], i .. format)
        local img = image.load(filename)
        data[(i-1) * 2 + j] = img
        label[(i-1) * 2 + j] = j
        if verbose then
            print(filename, (i-1) * 2 + j, j)
        end
    end
end

fullset.data = data
fullset.label = label

print('\nsaving', opt.savename)
torch.save(opt.savename, fullset)
