import numpy
import os
import torch
import math
import numpy
import PIL
from PIL import Image
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
from tqdm import tqdm
val_PATH = "T1/JPEGImages/"
val_RBG_PATH = "R1/RGB/"
ckpt_PATH = "Ckpt_Generator/Generator11.tar"#Enter Checkpoint path here
Image_PATH= "Generator_Output/" 
batch_size = 1
seed = 64
units = 64
alpha = 0.2
gray_max=2**8-1
x = os.listdir(val_PATH)
y = os.listdir(val_RBG_PATH)
x.sort()
y.sort()
length = len(x)
assert len(x) == len(y)
print("Length of x is equal to y ")
mse = nn.MSELoss()
L1 = nn.L1Loss()
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

def tensor(imag):
    imag = torch.from_numpy(imag)
    return imag

def save_images(tensor,name):
    device = torch.device("cpu")
    tensor = tensor.to(device)
    array = tensor.detach().numpy()
    array = numpy.squeeze(array)
    array = array*gray_max
    array = numpy.clip(array,0,255)
    array = numpy.uint8(array)
    array = numpy.rollaxis(array,0,3)
    array = Image.fromarray(array)
    array.save(Image_PATH+str(name)+".png")

device = torch.device("cuda:0")
class Generator(nn.Module):
    def __init__(self, units):
        super(Generator, self).__init__()
        #unit1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
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
        self.conv7 = nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1,padding_mode='zeros')

    def forward(self, x):
        global alpha
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


G = Generator(units).to(device)
G.load_state_dict(torch.load(ckpt_PATH))
G.eval()
print("Generator model has been instantiated")
print("The weights have been loaded ")
print("And set in evaluation mode")
print("All the best the with the images output ")

def PSNR(true,fake):
    mean_squared_error=mse(true,fake)
    mean_squared_error = mean_squared_error.item()
    db = 20*math.log10(gray_max/(mean_squared_error**0.5))*1/3
    return db


PSN=0
L1_Loss=0
for epoch in tqdm(range(length)):
    thermal_images,colour_images = get_images(location=epoch,thermal=x,colour=y,path=val_PATH,path1=val_RBG_PATH)
    thermal_images = thermal_images.to(device)
    colour_images = colour_images.to(device)
    output = G(thermal_images)
    save_images(tensor=output,name=epoch)
    PSN += PSNR(output,colour_images)
    L11  = L1(output,colour_images)
    L1_Loss += L11.item()
    
PSNR = PSN/(length)
print("Average PSNR is",PSNR,"dB")
print("Average L1 loss is",L1_Loss)