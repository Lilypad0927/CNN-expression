# -*- coding:utf-8 -*- 

from skimage import io,data
from PIL import Image
text = open('bbox.txt')
for line in text.readlines():
    content = line.split()
    Path = content[0]
    Path = 'image/' + Path[6] + '/' + Path[8:]
    #print Path
    Left = int(content[1])
    #print Left
    Right = int(content[2])
    #print Right
    Top = int(content[3])
    #print Top
    Bottom = int(content[4])
    #print Bottom
    img = io.imread(Path)
    roi = img[Top:Bottom,Left:Right]
    io.imsave(Path,roi)
#img = io.imread(Path)
#roi = img[177:395,268:486]  #top bottom left right
#io.imsave(Path,roi)