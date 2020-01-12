import PIL
import os 
from PIL import Image
from tqdm import tqdm
PATH = "JPEGImages/"
PATH_RGB = "RGB/"
Val_PATH ="Val_Thermal/"
Val_PATH_RGB = "Val_RGB/" 
PATH_1 = 'Thermal_600x500/Train/'
PATH_1_1 = 'Thermal_600x500/Val/'
PATH_2 = 'RGB_640_512/'
PATH_2_1 = 'Val_RGB_640_512/'
PATH_3 = 'Colour_1800x1500/Train/'
PATH_3_1 = 'Colour_1800x1500/Val/'
w = os.listdir("JPEGImages/")
x = os.listdir("RGB")
y = os.listdir("Val_Thermal")
z = os.listdir("Val_RGB")
w.sort()
x.sort()
y.sort()
z.sort()
length = len(x)
val_length = len(y)
print("Finished initializing paths and import folders")
def resize(path,root,width,height):
	imag = Image.open(path+root)
	imag = imag.resize((width,height))
	return imag

for i in tqdm(range(0,length)):
	#imag1 = resize(PATH,w[i],600,500)
	imag2 = resize(PATH_RGB,x[i],640,512)
	#imag3 = resize(PATH_RGB,x[i],1800,1500)
	#c = w[i]
	#c = c[:-5]
	d = x[i]
	d = d[:-4]
	#imag1.save(PATH_1+c+".png")
	imag2.save(PATH_2+d+".jpeg")
	#imag3.save(PATH_3+d+".png")
	if(i%1000==0):
		print("Finished saving 1,000 files ")

for i in tqdm(range(0,val_length)):
	#imag1 = resize(Val_PATH,y[i],600,500)
	imag2 = resize(Val_PATH_RGB,z[i],640,512)
	#imag3 = resize(Val_PATH_RGB,z[i],1800,1500)
	#c = y[i]
	#c = c[:-5]
	d = z[i]
	d = d[:-4]
	#imag1.save(PATH_1_1+c+".png")
	imag2.save(PATH_2_1+d+".jpeg")
	#imag3.save(PATH_3_1+d+".png")
	if(i%500==0):
		print("Finished saving 500 files")