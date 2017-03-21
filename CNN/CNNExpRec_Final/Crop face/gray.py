from skimage import io,data,color
text = open('bbox.txt')
for line in text.readlines():
    content = line.split()
    Path = content[0]
    Path = 'image/' + Path[6] + '/' + Path[8:]
    Path2 = 'gray_image/' + Path[6] + '/' + Path[8:]
    img=io.imread(Path)
    img_gray=color.rgb2gray(img)
    rows,cols=img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i,j]<=0.25):
                img_gray[i,j]=0
    io.imsave(Path2,img_gray)
