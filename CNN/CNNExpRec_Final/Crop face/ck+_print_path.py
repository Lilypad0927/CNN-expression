import os 
root = 'E:\extended-cohn-kanade-images\cohn-kanade-images'

def GetFileList(dir, fileList): 
    newDir = dir 
    if os.path.isfile(dir): 
        fileList.append(dir.decode('gbk')[(len(root)+1):]) 
    elif os.path.isdir(dir):   
        for s in os.listdir(dir): 
            newDir=os.path.join(dir,s) 
            GetFileList(newDir, fileList)   
    return fileList 

list = GetFileList(root, []) 

file = open('ck+ imagelist.txt','w')

print >> file, len(list);

for e in list: 
	print >> file, e;
	
file.close