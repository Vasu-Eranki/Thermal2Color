import numpy
import os
import torch
import math
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
import math
val_PATH = "Val_Thermal/"
val_RBG_PATH = "Val_Colour/"
ckpt_PATH = "Ckpt_CGAN/"#Enter Checkpoint path here
Image_PATH= "CGAN_Output/" 
batch_size = 1
seed = 64
units = 64
alpha = 0.2
mse = nn.MSELoss()
L1 = nn.L1Loss()
transform = transforms.Compose([transforms.Resize([640,512]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

thermal_val_dataset = datasets.ImageFolder(root=val_PATH, transform=transform)
colour_val_dataset = datasets.ImageFolder(root=val_RBG_PATH, transform=transform)
thermal_val = torch.utils.data.DataLoader(thermal_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
colour_val = torch.utils.data.DataLoader(colour_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
device = torch.device("cuda:0")
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
G = Generator(units).to(device)
G.load_state_dict(torch.load(ckpt_PATH))
G.eval()

PSNR=0
L1_Loss=0
for epoch in range(len(thermal_val)):
    thermal_images, _ = next(iter(thermal_val))
    colour_images, _ = next(iter(colour_val))
    thermal_images = thermal_images.to(device)
    colour_images = colour_images.to(device)
    output = G(thermal_images)
    PSN = mse(output,colour_images)
    L11  = L1(output,colour_images)
    L1_Loss += L11.item()
    PSN = PSN.item()
    PSNR += 20*math.log10(255/(PSN**0.5))*1/3
    save_image(output,Image_PATH+str(epoch)+".png")
PSNR = PSNR/(len(thermal_val))
print("Average PSNR is",PSNR,"dB")
print("Average L1 loss is",L1_Loss)
exit(0)