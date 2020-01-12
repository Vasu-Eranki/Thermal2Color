import os
import PIL
import math
import numpy
import torch
from PIL import Image
from torch import nn,optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision.utils import save_image
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt 

PATH = "JPEGImages/"  # Path for Thermal Images
PATH_RGB = "RGB/"  # Path for RGB Equivalent of Thermal Images
val_PATH = "Val_Thermal/"
val_RBG_PATH = "Val_RGB/"
Image_PATH= "Generator_Output/"
RGB_channels = 3
Gray_channels = 1
batch_size = 4
seed = 64
alpha = 0.2
units = 64
lr = 2e-4
beta1 = 0.9
beta2 = 0.999
gray_max = 2**8-1
w = os.listdir(PATH)
x = os.listdir(PATH_RGB)
y = os.listdir(val_PATH)
z = os.listdir(val_RBG_PATH)
length = len(x)
val_length = len(y)
epochs = math.ceil(100000/length)
w.sort()
x.sort()
y.sort()
z.sort()

def get_images(location,thermal,colour,path,path1):
    temp = thermal[location]
    temp1 = colour[location]
    img1 = image_open(path+temp)
    img2 = image_open(path1+temp1)
    img1 = image_resize(img1)
    img2 = image_resize(img2)
    img1 = image_to_array(img1,rgb=False)
    img2 = image_to_array(img2,rgb=True)
    img1 = normalisation(img1)
    img2 = normalisation(img2)
    img1 = tensor(imag=img1)
    img2 = tensor(imag=img2)
    return img1,img2

def image_resize(imag):
    imag = imag.resize((640,512))
    return imag

def image_open(path):
    image = Image.open(path)
    return image

def image_to_array(imag,rgb):
    imag = numpy.asarray(imag,dtype=numpy.float32)
    if rgb:
        imag = numpy.rollaxis(imag,2)
    else:
        imag = numpy.reshape(imag,(-1,640,1))
        imag = numpy.rollaxis(imag,2)
    imag = numpy.expand_dims(imag,0)
    return imag

def normalisation(imag):
    imag = imag/gray_max
    return imag

def tensor(imag): imag = torch.from_numpy(imag) return imag

print('Finished defining the datasets , thermal and colour datasets')
class Generator(nn.Module):
    def __init__(self, units):
        super(Generator, self).__init__()
        #unit1
        self.conv1 = nn.Conv2d(in_channels=Gray_channels, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn0   = nn.BatchNorm2d(32)
        #unit2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(64)
        #unit3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(128)
        #unit4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(128)
        # Defining Decoder
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        self.convt1 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bnt1 = nn.BatchNorm2d(192)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convt2 = nn.Conv2d(in_channels=256 , out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bnt2 = nn.BatchNorm2d(96)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode='zeros')
        self.bnt3  = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.conv7 = nn.Conv2d(in_channels=32,out_channels=RGB_channels,kernel_size=3,stride=1,padding=1,padding_mode='zeros')

    def forward(self, x):
        unit1 = F.leaky_relu(self.bn0(self.conv1(x)), alpha)
        unit2 = F.leaky_relu(self.bn1(self.conv2(unit1)), alpha)
        unit3 = F.leaky_relu(self.bn2(self.conv3(unit2)), alpha)
        unit4 = F.leaky_relu(self.bn3(self.conv4(unit3)), alpha)
        #Decoder
        x = F.relu(self.dropout1(self.bn4(self.conv5(unit4))))
        x = torch.cat((x,unit3),1)
        x = F.relu(self.dropout2(self.bnt1(self.convt1(x))))
        x = torch.cat((x,unit2),1)
        x = self.up(x)
        x = F.relu(self.dropout3(self.bnt2(self.convt2(x))))
        x = torch.cat((x,unit1),1)
        x = self.up(x)
        x = F.relu(self.dropout4(self.bnt3(self.conv6(x))))
        x = self.conv7(x)
        return x


print("Generator model has been declared")
device = torch.device("cuda:0")
G = Generator(units).to(device)
print(G)
print(summary(G, input_size=(Gray_channels,512,640)))
loss = nn.L1Loss()
mse  = nn.MSELoss()

def PSNR(true,fake):
    mean_squared_error=mse(true,fake)
    mean_squared_error = mean_squared_error.item()
    db = 20*math.log10(gray_max/(mean_squared_error**0.5))*1/3
    return db


Goptimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

# scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(Goptimizer,mode='min',factor=0.1,patience=int(epochs/10),verbose=True)

for epoch in range(epochs):
    step = 0
    pbar = tqdm(total=length)
    while(step<length):
        for a in range(0,batch_size):
            gray_images,rgb_images = get_images(location=step,thermal=w,colour=x,path=PATH,path1=PATH_RGB)
            gray_images = gray_images.to(device)
            rgb_images = rgb_images.to(device)
            G.zero_grad()
            fake = G(gray_images)
            errorG = loss(fake,rgb_images)
            errorG.backward()
            step+=1
            pbar.update(1)
            # Now training the generator
        Goptimizer.step()
        # scheduler1.step(errorG)
    #Saving the checkpoint file at every epoch
    print('[%d/%d] Loss_G: %.4f'% (epoch + 1, epochs,errorG.item()))
    torch.save(G.state_dict(),"Ckpt_Generator/Generator"+str(epoch)+".tar")
    psnr=0
    pbar.close()
    for epoch in range(0,val_length):
        thermal_images,colour_images = get_images(location=epoch,thermal=y,colour=z,path=val_PATH,path1=val_RBG_PATH)
        thermal_images = thermal_images.to(device)
        colour_images = colour_images.to(device)
        output = G(thermal_images)
        psnr += PSNR(true=colour_images,fake=output)
    psnr = psnr/val_length
    print("Average PSNR is %.4f"%psnr)

print("Training and checkpoint files have been saved")
