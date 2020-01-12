import numpy
import os
import torch
import skimage
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
PATH = "Thermal/"  # Path for Thermal Images
PATH_RGB = "Colour/"  # Path for RGB Equivalent of Thermal Images
val_PATH = "Val_Thermal/"
val_RBG_PATH = "Val_Colour/"
width = 256
height = 256
RGB_channels = 3
Gray_channels = 1
tchannels = RGB_channels+Gray_channels
batch_size = 1
seed = 64
alpha = 0.2
units = 64
lambda_L1 = 100
true = 0.9
false = 0.1
epochs = 12
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
colour_max = 2**8-1
gray_max = 2**8-1
w = os.listdir(PATH)
x = os.listdir(PATH_RGB)
y = os.listdir(val_PATH)
z = os.listdir(val_RBG_PATHl)
length = len(x)
val_length = len(y)
epochs = math.ceil(1000000/length)
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

def tensor(imag):
    imag = torch.from_numpy(imag)
    return imag

# Finished defining the datasets , thermal and colour datasets


class Discriminator(nn.Module):
    def __init__(self, units):
        super(Discriminator, self).__init__()

    # Defining the Discriminator_Channels
        self.conv1 = nn.Conv2d(in_channels=tchannels, out_channels=units, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        #128X128 with units channels
        self.conv2 = nn.Conv2d(in_channels=units, out_channels=units * 2, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(units * 2)
        #64x64 with 2*units channels
        self.conv3 = nn.Conv2d(in_channels=units * 2, out_channels=units * 4, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(units * 4)
        #32x32 with 4 units channels
        self.conv4 = nn.Conv2d(in_channels=units * 4, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(units * 8)
        #16x16 with 8 units channels
        self.pad = nn.ZeroPad2d((0, 3, 0, 3))
        #After padding it becomes 19x19
        self.conv5 = nn.Conv2d(in_channels=units * 8, out_channels=units * 8, kernel_size=4, stride=1, padding=0, padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(units * 8)
        #16x16 with units*8channels 
        #Becomes 19x19 with padding once again 
        #Becomes 16x16 with 1 channel 
        self.conv6 = nn.Conv2d(in_channels=units * 8, out_channels=1, kernel_size=4, stride=1, padding=0, padding_mode='zeros')

    def forward(self, x):
        global alpha
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.bn1(self.conv2(x)), alpha)
        x = F.leaky_relu(self.bn2(self.conv3(x)), alpha)
        x = F.leaky_relu(self.bn3(self.conv4(x)), alpha)
        x = self.pad(x)
        x = F.leaky_relu(self.bn4(self.conv5(x)), alpha)
        x = self.pad(x)
        x = torch.sigmoid(self.conv6(x))
        return x


print("Discriminator Model has been declared")
def weights_initialization()


class Generator(nn.Module):
    def __init__(self, units):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=RGB_channels, out_channels=units, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=units, out_channels=units * 2, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(units * 2)
        self.conv3 = nn.Conv2d(in_channels=units * 2, out_channels=units * 4, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(units * 4)
        self.conv4 = nn.Conv2d(in_channels=units * 4, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(units * 8)
        self.conv5 = nn.Conv2d(in_channels=units * 8, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(units * 8)
        self.conv6 = nn.Conv2d(in_channels=units * 8, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn5 = nn.BatchNorm2d(units * 8)
        self.conv7 = nn.Conv2d(in_channels=units * 8, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bn6 = nn.BatchNorm2d(units * 8)
        self.conv8 = nn.Conv2d(in_channels=units * 8, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        # Defining Decoder
        self.convt1 = nn.ConvTranspose2d(in_channels=units * 8, out_channels=units * 8, kernel_size=[5,4], stride=2, padding=1, padding_mode='zeros')
        self.bnt1 = nn.BatchNorm2d(units * 8)
        self.convt2 = nn.ConvTranspose2d(in_channels=units * 16, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt2 = nn.BatchNorm2d(units * 8)
        self.convt3 = nn.ConvTranspose2d(in_channels=units * 16, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt3 = nn.BatchNorm2d(units * 8)
        self.convt4 = nn.ConvTranspose2d(in_channels=units * 16, out_channels=units * 8, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt4 = nn.BatchNorm2d(units * 8)
        self.convt5 = nn.ConvTranspose2d(in_channels=units * 16, out_channels=units * 4, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt5 = nn.BatchNorm2d(units * 4)
        self.convt6 = nn.ConvTranspose2d(in_channels=units * 8, out_channels=units * 2, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt6 = nn.BatchNorm2d(units * 2)
        self.convt7 = nn.ConvTranspose2d(in_channels=units * 4, out_channels=units, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        self.bnt7 = nn.BatchNorm2d(units)
        self.convt8 = nn.ConvTranspose2d(in_channels=units * 2, out_channels=RGB_channels, kernel_size=4, stride=2, padding=1, padding_mode='zeros')

    def forward(self, x):
        global alpha
        unit1 = F.leaky_relu(self.conv1(x), alpha)
        unit2 = F.leaky_relu(self.bn1(self.conv2(unit1)), alpha)
        unit3 = F.leaky_relu(self.bn2(self.conv3(unit2)), alpha)
        unit4 = F.leaky_relu(self.bn3(self.conv4(unit3)), alpha)
        unit5 = F.leaky_relu(self.bn4(self.conv5(unit4)), alpha)
        unit6 = F.leaky_relu(self.bn5(self.conv6(unit5)), alpha)
        unit7 = F.leaky_relu(self.bn6(self.conv7(unit6)), alpha)
        unit8 = F.relu(self.conv8(unit7))
        x = F.relu(self.bnt1(self.convt1(unit8)))
        x = torch.cat((x, unit7), 1)
        x = F.relu(self.bnt2(self.convt2(x)))
        x = torch.cat((x, unit6), 1)
        x = F.relu(self.bnt3(self.convt3(x)))
        x = torch.cat((x, unit5), 1)
        x = F.relu(self.bnt4(self.convt4(x)))
        x = torch.cat((x, unit4), 1)
        x = F.relu(self.bnt5(self.convt5(x)))
        x = torch.cat((x, unit3), 1)
        x = F.relu(self.bnt6(self.convt6(x)))
        x = torch.cat((x, unit2), 1)
        x = F.relu(self.bnt7(self.convt7(x)))
        x = torch.cat((x, unit1), 1)
        x = torch.tanh(self.convt8(x))
        return x


print("Generator model has been declared")
device = torch.device("cuda:0")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = Generator(units).to(device)
D = Discriminator(units).to(device)
print(G)
print(D)
print("-----------------------------------------------------------------------------------------------------")
print(summary(G, input_size=(Gray_channels, 640, 512)))
print(summary(D, input_size=(tchannels, 640, 512)))
print("-----------------------------------------------------------------------------------------------------")
loss = nn.BCELoss()
L1_Loss = nn.L1Loss()

Goptimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
Doptimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))


# scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(Goptimizer,mode='min',factor=0.1,patience=int(epochs/10),verbose=True)
# scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(Doptimizer,mode='min',factor=0.1,patience=int(epochs/10),verbose=True


for epoch in range(epochs):
    step = 0
    while(step<length):
        for a in range(0,batch_size):
            gray_images,rgb_images = get_images(location=step,thermal=w,colour=x,path=PATH,path1=PATH_RGB)
            gray_images = gray_images.to(device)
            rgb_images = rgb_images.to(device)
            batch_size = gray_images.size(0)
            input_images = torch.cat((rgb_images, gray_images), 1)
            D.zero_grad()
            labels = torch.full((batch_size, 1, 40, 32), true, device=device)
            output = D(input_images)
            errorD = loss(output, labels)*0.5
            errorD.backward()
            D_x = output.mean().item()
            # Train with noise
            fake = G(gray_images)
            fake1 = torch.cat((fake,gray_images),1)
            output = D(fake1.detach())
            D_G = output.mean().item()        
            labels.fill_(false)
            errorD_1 = loss(output, labels)*0.5
            errorD_1.backward()
            D_x_1 = output.mean().item()
        Doptimizer.step()
            # Finished training the Discriminator
            # Now training the generator
        for a in range(0,batch_size):
            G.zero_grad()
            labels.fill_(true)
            output = D(fake1)
            D_G_1 = output.mean().item()
            errorG = loss(output, labels)
            errorG.backward(retain_graph=True)
            errorG_L1_Loss = 100 * L1_Loss(fake, rgb_images)
            errorG_L1_Loss.backward()
        Goptimizer.step()
        # scheduler1.step(errorG)
        # scheduler2.step(errorD)
        step+=batch_size
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f '% (epoch + 1, epochs,errorD.item(), errorG.item(), D_x, D_G, D_G_1))
    #Saving the checkpoint file at every epoch 
    torch.save({
        'epoch':epoch,
        'model_state_dict':G.state_dict(),
        'optimizer_state_dict':Goptimizer.state_dict(),
        'loss':(errorG,errorG_L1_Loss)
        },"Ckpt_CGAN/Generator_"+str(epoch)+".tar")    
    torch.save({
        'epoch':epoch,
        'model_state_dict':D.state_dict(),
        'optimizer_state_dict':Doptimizer.state_dict(),
        'loss':(errorD,errorD_1)
        },"Ckpt_CGAN/Discriminator_"+str(epoch)+".tar")
    
    psnr=0
    for epoch in range(0,val_length):
        thermal_images,colour_images = get_images(location=epoch,thermal=y,colour=z,path=val_PATH,path1=val_RBG_PATH)
        thermal_images = thermal_images.to(device)
        colour_images = colour_images.to(device)
        output = G(thermal_images)
        psnr += PSNR(true=colour_images,fake=output)
    psnr = psnr/val_length
    print("Average PSNR is %.4f"%psnr)  
    print("Average PSNR for this",epoch,"is",PSNR,"dB")