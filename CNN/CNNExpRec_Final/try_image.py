import numpy as np
from scipy.misc import imresize
import matplotlib.image
import matplotlib.pyplot

image = matplotlib.image.imread('data/jaffe_images_small/NA.SA2.206.tiff')
print image
#matplotlib.pyplot.imshow(image)
image = image.tolist()
image = imresize(image, (48,48))
print image
matplotlib.pyplot.imshow(image)