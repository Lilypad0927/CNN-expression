import os 

# -------------- print label list -------------------
'''
root = 'Emotion_labels\Emotion'
def GetFileList(dir, fileList): 
    newDir = dir 
    if os.path.isfile(dir): 
		if '.DS_Store' in dir:
			print '.DS_Store'
		else:
			fileList.append(dir[:]) 
    elif os.path.isdir(dir):   
        for s in os.listdir(dir): 
            newDir=os.path.join(dir,s) 
            GetFileList(newDir, fileList)   
    return fileList 

list = GetFileList(root, []) 
file = open('label_list.txt','w')
for e in list: 
	print >> file, e;
file.close
'''
# --------------------------------------------

# -------------- print image list -------------------
'''
root = 'extended-cohn-kanade-images'
def GetFileList(dir, fileList): 
    newDir = dir 
    if os.path.isfile(dir): 
		if '.DS_Store' in dir:
			print '.DS_Store'
		else:
			fileList.append(dir[:]) 
    elif os.path.isdir(dir):   
        for s in os.listdir(dir): 
            newDir=os.path.join(dir,s) 
            GetFileList(newDir, fileList)   
    return fileList 

list = GetFileList(root, []) 
file = open('image_list.txt','w')
for e in list: 
	print >> file, e;
file.close
'''
# --------------------------------------------

# -------------- select images ---------------
'''
images = open('image_list.txt')
last = 'extended-cohn-kanade-images\cohn-kanade-images\S005\\001\S005_001_00000001.png'
list = []
result = []
for line in images.readlines():
	if last[0:55] != line[0:55]:
		list = list [len(list)/2:]
		file = open('select_list.txt','w')
		for e in list: 
			result.append(e)
		list = []
		last = line
	list.append(line[0:len(line)-1])
file = open('select_list.txt','w')
for e in result: 
	print >> file, e;
file.close
'''
# -----------------------------------------

import shutil
text = open('label_list.txt')
i = 0
for line in text.readlines():
	file = open(line[0:len(line)-1])
	s = file.readline()
	label = s[3]
	images = open('select_list.txt')
	for image in images.readlines():
		'''
		i = i+1
		print image[48:51]
		print line[24:27]
		print image[52:55]
		print line[28:31]
		print
		if i > 10:
			break
		'''
		if image[48:51] == line[24:27] and image[52:55] == line[28:31]:
			newfile = 'CK+ extend\\' + label + image[55:len(image)-1]
			shutil.copyfile(image[0:len(image)-1],newfile)
		images.close()
	continue

'''
for file in sorted(os.listdir(root)) :
	if (file != '.DS_Store'):
text = open('bbox.txt')
for line in text.readlines():
'''