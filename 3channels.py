import os 
import numpy 
import PIL 
from PIL import Image 
import tqdm 
from tqdm import tqdm 
PATH = "Jpeg_1/"
PATH_1="RGB_640_512/"
FINAL_PATH="paired"
#PATH="T1/JPEGImages/"
x=os.listdir(PATH)
y=os.listdir(PATH_1)
x.sort()
y.sort()
assert len(x)==len(y)
print("Both folders have the same number of images")
for i in tqdm(range(0,len(x))):
	img1 = Image.open(PATH+x[i])
	img2 = Image.open(PATH_1+y[i])
	img1= numpy.asarray(img1)
	img1 = numpy.stack((img1,)*3,axis=-1)
	img2 = numpy.asarray(img2)
	combine = numpy.hstack((img1,img2))
	combine = numpy.uint8(combine)
	combine = Image.fromarray(combine)
	combine.save("paired"+x[i])
'''
for i in tqdm(range(0,len(x))):
	img1 = Image.open(PATH+x[i])
	img1 = numpy.asarray(img1)
	img1 = numpy.stack((img1,)*3,axis=-1)
	img1 = Image.fromarray(img1)
	img1.save("T11/"+x[i])