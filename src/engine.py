#!/usr/bin/env python
# coding=utf-8

from PIL import Image, ImageDraw, ImageFont
import random
import math
import os
import argparse

def operation(x, op, y, mode):
    '''
    return equation string giving x, y, operator, within mode of 'chi' or 'en'
    '''
    lang_mapper = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
    opt_mapper_en = {'+':'加上', '-':'减去', '*':'乘以', '/':'除以'}
    opt_mapper_chi = {'+':'加', '-':'减', '*':'乘'}
    symbol = '？'
    if mode == 'chi':
        x = lang_mapper[x]
        op = opt_mapper_chi[op]
        y = lang_mapper[y]
        symbol = ''
    elif mode == 'en':
        x = str(x)
        op = opt_mapper_en[op]
        y = str(y)
    ret = x + op + y + '等于' + symbol
    return ret.decode('utf-8')

def randomEquation(mode='en'):
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    op = random.choice(['+', '-', '*', '/'])
    if mode == 'chi':           # no '/' for mode = 'chi'
        op = random.choice(['+', '-', '*'])
    # constrains for en
    elif op == '-':             # x is always bigger for x - y for en mode
        if x < y:
            temp = x
            x = y
            y = temp
    elif op == '/':             # to make sure (x % y == 0) and (y != 0)
        y = random.randint(1, 9)
        x = x * y
    equation = operation(x, op, y, mode)
    return equation

ChiSayings = []
def randomChiSaying():
    '''
    generate a random chinese saying from txt
    '''
    if len(ChiSayings) == 0:        # load data first
        filepath = '../synpic/chisayings.txt'
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = unicode(line.strip(), 'utf-8')
                ChiSayings.append(line)
        print(str(len(ChiSayings)) + ' Chinese Sayings detected!')
    index = random.randint(0, len(ChiSayings))
    return ChiSayings[index]

'''
generate a random image with Class ImageGenerator

example:
    from generator import ImageGenerator
    ig = ImageGenerator()
    ig.generateImage(string='1776', path='./1.jpg')
'''
class ImageGenerator(object):
    def __init__(self, fontPath, fontSize=24, mag=1, size = (200, 50), bgColor = (200, 200, 200)):
        '''
        declare and initialize private varians
        '''
        self.mag = mag
        self.size = size
        self.fontPath = fontPath
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
    pass

    def randRGB(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    pass

    def randPoint(self, num=200, color=-1):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            x = random.randint(0, self.size[0])
            y = random.randint(0, self.size[1])
            if color < 0:
                draw.point((x, y), fill=self.randRGB())
            else:
                draw.point((x, y), fill=(color, color, color))
            pass
        pass
    pass

    def randLine(self, num=30, length=12, color=100):
        '''
        make some random line noises
        '''
        length = length + random.randint(-length / 4, length / 4)
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            # draw a line from random point(x1, y1) at random angle
            x1 = random.randint(0, self.size[0])
            y1 = random.randint(0, self.size[1])
            angle = random.randint(0, 360) * math.pi / 180
            x2 = x1 + length * math.cos(angle)
            y2 = y1 + length * math.sin(angle)
            if color < 0: # mean randRGB
                draw.line([(x1, y1), (x2, y2)], self.randRGB())
            else:
                draw.line([(x1, y1), (x2, y2)], (color, color, color))
            pass
        pass

    pass

    def drawChar(self, text, angle=random.randint(-10, 10)):
        '''
        get a sub image with one specific character
        '''
        charImg = Image.new('RGBA', (int(self.fontSize * 1.3), int(self.fontSize * 1.3)))
        ImageDraw.Draw(charImg).text((0, 0), text, font=self.font, fill=self.randRGB())
        charImg = charImg.crop(charImg.getbbox())
        charImg = charImg.rotate(angle, Image.BILINEAR, expand=1)
        return charImg
    pass

    def generateImage(self, strings = u'8除以6等于？', path='out.jpg'):
        '''
        genreate a picture giving string and path
        '''
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randLine()
        gap = 2 # pixes between two characters
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-60, 60))
            x = start + self.fontSize * i + random.randint(0, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-10, 10)
            self.image.paste(charImg, (x, y), charImg)
        self.image = self.image.resize((int(self.size[0] * self.mag), int(self.size[1] * self.mag)), Image.BILINEAR)
        self.image.save(path)
    pass


class Type2ImageGenerator(ImageGenerator):
    def __init__(self, fontPath, fontSize=28, mag=1, size = (160, 53), bgColor = (255, 255, 255)):
        super(Type2ImageGenerator, self).__init__(fontPath, fontSize, mag, size, bgColor)
    pass

    def generateImage(self, strings = u'叁加陆等于', path='out.jpg'):
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randPoint()
        self.randLine(num=random.randint(15, 20), length=100, color=-1)
        gap = 4
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-15, 15))
            x = start + self.fontSize * i + random.randint(1, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-10, 10)
            self.image.paste(charImg, (x, y), charImg)
            self.image = self.image.resize((int(self.size[0] * self.mag), int(self.size[1] * self.mag)), Image.BILINEAR)
            self.image = self.image.resize(self.size, Image.BILINEAR)
            self.image.save(path)
        pass
    pass

class Type3ImageGenerator(Type2ImageGenerator):
    def __init__(self, fontPath, fontSize=36):
        super(Type3ImageGenerator, self).__init__(fontPath, fontSize)

    def generateImage(self, strings = u'参差不齐', path='out.jpg'):
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randPoint()
        self.randLine(num=random.randint(15, 20), length=100, color=-1)
        gap = 5
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-15, 15))
            x = start + self.fontSize * i + random.randint(1, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-6, 6)
            self.image.paste(charImg, (x, y), charImg)
            self.image = self.image.resize((int(self.size[0] * self.mag), int(self.size[1] * self.mag)), Image.BILINEAR)
            self.image = self.image.resize(self.size, Image.BILINEAR)
            self.image.save(path)
        pass
    pass

def syntheticData(args):
    if type(args.fonts) == str:
        args.fonts = [str]
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
        print('mkdir ' + args.savedir)
    print('enter ' + args.savedir)
    for font in args.fonts:
        '''
        generate data for every font
        '''
        if args.verbose == True:
            print('reading ' + font)
        ig = []
        if args.type == 1:
            ig = ImageGenerator(font)
        elif args.type == 2:
            ig = Type2ImageGenerator(font)
        elif args.type == 3:
            ig = Type3ImageGenerator(font)
        fontname = font.split('.')[-2].split('/')[-1]
        for i in range(args.number):
            if args.number > 100 and i % (args.number / 10) == 0:
                print i*100/args.number, "%..."
            txtpath = os.path.join(args.savedir, fontname + str(i) + '.gt.txt')
            with open(txtpath, 'w') as f:
                label = ''
                if args.type == 1:
                    label = randomEquation(mode='en')
                elif args.type == 2:
                    label = randomEquation(mode='chi')
                elif args.type == 3:
                    label = randomChiSaying()
                f.write(label.encode('utf-8')+ '\n')
                filepath = os.path.join(args.savedir, fontname + str(i) + '.jpg')
                ig.generateImage(strings=label, path=filepath)
            pass
        pass
    pass

def getAllFonts(fontdir = '../fonts/'):
    fonts = []
    for font in os.listdir(fontdir):
        ext = font.split('.')[1]
        if ext == 'ttf' or ext == 'TTF' or ext == 'ttc':
            fonts.append(os.path.join(fontdir, font))
    print str(len(fonts)) + ' fonts detected!'
    return fonts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default=1, type=int, help="which type of CAPTCHA to generate")
    parser.add_argument("-d", "--savedir", default="../synpic/temp/", help="directory to save the pictures")
    parser.add_argument("-f", "--fonts", default=getAllFonts(), help="choose which font to use")
    parser.add_argument("-n", "--number", default=2, type=int, help="how many pictures to generate for every font?")

    feature = parser.add_mutually_exclusive_group(required=False)
    feature.add_argument("--verbose", dest='verbose', action='store_true', help="print verbose infomation")
    feature.add_argument("--no-verbose", dest='verbose', action='store_false', help="forbid verbose infomation")
    parser.set_defaults(feature=True)

    args = parser.parse_args()
    #print(args)
    syntheticData(args)
