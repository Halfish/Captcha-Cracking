## Captcha-Cracking Program Using Torch

This is a program aiming to crack some CAPTCHA on several website, which may
include both traditional method as well as deep learning method.

**1. Traditional Methods**

By traditional methods, we firstly need to preprocess the image like removing noises
in the background, and do the slant correction if the character have some rotated angles.
Then, cut out each single characters and train a classifier to recognize them.

**2. Deep Learning Methods**

In this program, we mainly use a Convolutional Neural Network model developed by Google,
which was firstly desigined to extract street view house number(SVHN) from Google Map.
Click here to read the origin article.
[Multi-digi Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/abs/1312.6082)

## About Torch7 & OpenCV
[Torch7](http://torch.ch) is a scientific computing framework based on Lua. We can easily build
any complex deep learing model using Torch7.
Install torch7 by following these commands,
```shell
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps
./install.sh
source ~/.zshrc
```
[OpenCV](http://opencv.org) is a open source computer vision library. We use opencv to pre-process
the image before we formally begin the recogize it. And we mainly use Python interface in the program.
Install opencv through apt-get
```shell
sudo apt-get install python-opencv
```

## web service
[Tornado](http://www.tornadoweb.org/en/stable/index.html) is a Python web framewrok and asynchronous networking library.
Install tornado-4.3 by pip, and using redis to connect tornado and torch7.
```shell
sudo apt-get install tesseract-ocr-chi-sim
sudo pip install pytesseract
sudo pip install futures
sudo pip install tornado
sudo apt-get install redis-server
redis-server &
sudo pip install redis      # install redis for python interface
luarocks install redis-lua  # install redis for lua interface

# start service
git clone https://github.com/Halfish/Captcha-Cracking.git
cd captcha-web
nohup python upload.py & > logpy.txt
nohup th breaking.lua -gpuid 1 & > loglua.txt
```

## Model A: SVHN Model
When cracking type1 to type6 CAPTCHA, our model is always prefixed with svhn, 
which we have already explained what is SVHN up there, because this model is used to recognize SVHN at first.
The details are listed as following.

Step 0: Go to the ./src/ sub derectory
```shell
cd src/
```

Step 1: Generate synthetic pictures with labels
```python
# to see how to use engine
python engine.py -h

# generate 1000 type 2 pictures, saving in ../trainpic/type2/
python engine.py -t 2 -n 1000 -d ../trainpic/type2
python engine.py -t 6 -n 1000 -d ../trainpic/type6
```

Step 2: Dump full data set
```lua
-- to see how to dump data
th svhn_dump.lua -h

-- dump 1000 type 2 picture for every font
th svhn_dump.lua -persize 1000 -type 2 -datadir ../trainpic/type2 -savename type2_1000.dat
th svhn_dump.lua -persize 1000 -type 6 -datadir ../trainpic/type6 -savename type6_1000.dat
```

Step 3: Train the model
```lua
-- to see how to train a model
th svhn_train.lua -h

-- using GPU-2(start from 1) to train a CNN model from type1 CAPTCHA
th svhn_train.lua -gpuid 2 -type 1 -dataname type1_data.dat -savename model_type1.t7
th svhn_train.lua -gpuid 2 -type 6 -dataname type6_1000.dat -savename model_type6.t7
```

## Model B: Simple Model
Some type of Captcha has fixed position of every character we need to crack, so we can cut out and
use any simple classifier to recognize them. But the pre-process is essential and important.
Our type4 Captcha, including four websites belonging to four provinces, can be cracked by this way.
Type4 Captcha including chongqing(chq), gansu(gs), ningxia(nx) and tianjin(tj).
Here are some details.

Step 0: 
```shell
cd src/
```

Step 1: Generate some pictures with labels
```shell
python type4_cutAndDump.py chq
python type4_cutAndDump.py gs
python type4_cutAndDump.py nx
python type4_cutAndDump.py tj
```
This script will generate some pictures under ./trainpic/type4/

Step 2: dump data before training
```shell
th type4_dump.lua -province chq -typename num
th type4_dump.lua -province chq -typename symb

th type4_dump.lua -province gs -typename num
th type4_dump.lua -province gs -typename symb

th type4_dump.lua -province nx -typename num
th type4_dump.lua -province nx -typename symb

th type4_dump.lua -province tj -typename num
th type4_dump.lua -province tj -typename symb
```
You can manually move the *.dat to ../data/ for better directory organization.

Step 3: training
```shell
th type4_train.lua -maxiters 300 -model chq -type num -datpath ../data/type4_chq_num.dat
```

Step 4: prediction
We have 200 pictures without labels prepared for prediction. 
Or you can just predict just one picture.
```shell
th type4_predict.py -province chq -picpath ../testpic/type4/chq/5000.png
th type4_predict.py -province chq 
```
