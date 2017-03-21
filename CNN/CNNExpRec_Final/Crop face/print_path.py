# -*- coding:utf-8 -*- 

import os 

root = '/home/ly/Crop face/image'

def GetFileList(dir, fileList): 
    newDir = dir 
    if os.path.isfile(dir): 
        fileList.append(dir[(len(root)+1):]) 
    elif os.path.isdir(dir):   
        for s in os.listdir(dir): 
            newDir=os.path.join(dir,s) 
            GetFileList(newDir, fileList)   
    return fileList 

list = GetFileList(root, []) 

file = open('imagelist.txt','w')

print >> file, len(list);

for e in list: 
	print >> file, e;
	
file.close
