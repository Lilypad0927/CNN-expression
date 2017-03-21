# _*_ coding:utf-8 _*_
import csv
from skimage import io,data
csvfile1 = file('csv1.csv', 'wb')
csvfile2 = file('csv2.csv', 'wb')
csvfile3 = file('csv3.csv', 'wb')
csvfile4 = file('csv4.csv', 'wb')
csvfile5 = file('csv5.csv', 'wb')
csvfile6 = file('csv6.csv', 'wb')
csvfile7 = file('csv7.csv', 'wb')
writer1 = csv.writer(csvfile1)
writer1.writerow(['emotion', 'pixels', 'Usage'])
writer2 = csv.writer(csvfile2)
writer2.writerow(['emotion', 'pixels', 'Usage'])
writer3 = csv.writer(csvfile3)
writer3.writerow(['emotion', 'pixels', 'Usage'])
writer4 = csv.writer(csvfile4)
writer4.writerow(['emotion', 'pixels', 'Usage'])
writer5 = csv.writer(csvfile5)
writer5.writerow(['emotion', 'pixels', 'Usage'])
writer6 = csv.writer(csvfile6)
writer6.writerow(['emotion', 'pixels', 'Usage'])
writer7 = csv.writer(csvfile7)
writer7.writerow(['emotion', 'pixels', 'Usage'])
Usage = 'Training'
text = open('Crop face2/bbox.txt')
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
flag1 = 3
flag2 = 3
flag3 = 3
flag4 = 3
flag5 = 3
flag6 = 3
flag7 = 3
for line in text.readlines():
    content = line.split()
    Path = content[0]
    emotion = Path[6]
    Path = 'Crop face2/gray image small/' + Path[6] + '/' + Path[8:]
    img = io.imread(Path)
    pixels = ''
    for line in img:
	for pixel in line:
	    pixels += str(pixel)
	    pixels += ' '
    if emotion == '1':
	emotion = '0'
	if flag1 > 0:
            data1.append((emotion,pixels,Usage))
	    flag1 = flag1 - 1
	else:
	    data1.append((emotion,pixels,'PublicTest'))
	    flag1 = 3
    elif emotion == '2':
	emotion = '6'
	if flag7 > 0:
	    data7.append((emotion,pixels,Usage))
	    flag7 = flag7 - 1
	else:
	    data7.append((emotion,pixels,'PublicTest'))
	    flag7 = 3
    elif emotion == '3':
	emotion = '1'
	if flag2 > 0:
	    data2.append((emotion,pixels,Usage))
	    flag2 = flag2 - 1
	else:
	    data2.append((emotion,pixels,'PublicTest'))
	    flag2 = 3
    elif emotion == '4':
	emotion = '2'
	if flag3 > 0:
	    data3.append((emotion,pixels,Usage))
	    flag3 = flag3 - 1
	else:
	    data3.append((emotion,pixels,'PublicTest'))
	    flag3 = 3
    elif emotion == '5':
	emotion = '3'
	if flag4 > 0:
	    data4.append((emotion,pixels,Usage))
	    flag4 = flag4 - 1
	else:
	    data4.append((emotion,pixels,'PublicTest'))
	    flag4 = 3
    elif emotion == '6':
	emotion = '4'
	if flag5 > 0:
	    data5.append((emotion,pixels,Usage))
	    flag5 = flag5 - 1
	else:
	    data5.append((emotion,pixels,'PublicTest'))
	    flag5 = 3
    elif emotion == '7':
	emotion = '5'
	if flag6 > 0:
            data6.append((emotion,pixels,Usage))
	    flag6 = flag6 - 1
	else:
	    data6.append((emotion,pixels,'PublicTest'))
	    flag6 = 3
writer1.writerows(data1)
writer2.writerows(data2)
writer3.writerows(data3)
writer4.writerows(data4)
writer5.writerows(data5)
writer6.writerows(data6)
writer7.writerows(data7)
csvfile1.close()
csvfile2.close()
csvfile3.close()
csvfile4.close()
csvfile5.close()
csvfile6.close()
csvfile7.close()

