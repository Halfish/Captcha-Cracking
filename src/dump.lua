require 'image';
require 'io';
require 'decoder'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-persize', 10, 'How many pictures to dump for every font?')
cmd:option('-datadir', '../synpic/type1/', 'Where to find pictures to dump?')
cmd:option('-savename', 'fullset.dat', 'save the dataset as ...')
cmd:text()
opt = cmd:parse(arg or {})

fonts = {}
for file in paths.iterfiles('../fonts/') do
    fonts[#fonts + 1] = string.split(file, '%.')[1]
end
print("Fonts:")
print(fonts)
print('')

fullset = {}
fullset.size = opt.persize * #fonts

data = torch.Tensor(fullset.size, 3, 50, 200)
labels = torch.IntTensor(fullset.size, 9):fill(0)
print(string.format("size of data:\n%s", #data))
print(string.format("size of labels:\n%s", #labels))
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
        labels[(i-1) * #fonts + j] = str2label(str)
    end
end

fullset.data = data
fullset.labels = labels

print("\nsaving", opt.savename)
torch.save(opt.savename, fullset)
