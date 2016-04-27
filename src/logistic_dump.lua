require 'image'
require 'io'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type', 1, 'Which type to dump')
cmd:option('-persize', 10, 'how many pictures to dump for every font')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:text()
local opt = cmd:parse(arg or {})

local fonts = {'simsun', 'STSONG'}
local dirs = {'../trainpic/type2', '../trainpic/type3'}

local fullset = {}
fullset.size = opt.persize * #fonts

local data, label
data = torch.Tensor(fullset.size, 3, 53, 160)
label = torch.IntTensor(fullset.size):fill(0)

for i = 1, opt.persize do
    for j = 1, #fonts do
        local filename = path.join(dirs[j], fonts[j] .. i-1 .. '.jpg')
        local img = image.load(filename)
        data[(i-1) * #fonts + j] = img
        label[(i-1) * #fonts + j] = j
    end
end

fullset.data = data
fullset.label = label

print('\nsaving', opt.savename)
torch.save(opt.savename, fullset)
