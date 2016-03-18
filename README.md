## 某网站验证码识别破解 (Captcha-Cracking Program Using Torch)

#### Step 0: Go to the ./src/ sub derectory
```shell
cd src/
```

#### Step 1: Generate synthetic pictures with labels
```python
# to see how to use engine
python engine.py -h

# generate 1000 type 2 pictures, saving in ../synpic/type2/
python engine.py -t 2 -n 1000 -d ../synpic/type2
```

#### Step 2: Dump full data set
```lua
-- to see how to dump data
th dump.lua -h

-- dump 1000 type 2 picture for every font
th dump.lua -persize 1000 -datadir ../synpic/type2 -savename type2_1000.dat
```

#### Step 3: Train the model
```lua
-- to see how to train a model
th svhn.lua -h

-- using GPU-2 to train a CNN model from type1 CAPTCHA
th svhn.lua -gpuid 2 -type 1 -dataname type1_data.dat -savename model_type1.t7
```
