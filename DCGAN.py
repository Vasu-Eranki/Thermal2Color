!pip3 install torchsummary
import numpy 
import pandas
import os 
import matplotlib.pyplot as plt 
import torch 
from torch import nn,optim 
import torch.nn.functional as F 
from torchvision import datasets , transforms 
from torchvision.utils import save_image
#from torchsummary import summary 
print(os.listdir("../input"))
print(os.listdir("../input/generative-dog-images/all-dogs/"))
import warnings 
warnings.filterwarnings("ignore")
batch_size=32

transform = transforms.Compose([transforms.Resize(64),
                                 transforms.CenterCrop(64),
                                 transforms.RandomRotation(degrees=45),   
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])
                          
train_data = datasets.ImageFolder("../input/generative-dog-images/all-dogs/",transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,batch_size=batch_size,num_workers=4,drop_last=True)
print("Hahaha done")
print(train_data)
print(train_loader)
class Generator(nn.Module):
    def __init__(self,nfeats,nchannels):
        super (Generator,self).__init__()
        #Feeding Latent vector which will be the dot product of two latent vectors of size 1X1X100
        self.convt1 = nn.ConvTranspose2d(in_channels=100,out_channels=nfeats*2,kernel_size=4,stride=1,padding=0,bias=False,padding_mode='zeros')
        #Output has the shape 4X4 with nfeats*2 channels which will be tentatively 128 channels if nfeats = 64
        self.convt2 = nn.ConvTranspose2d(in_channels=nfeats*2,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='zeros')
        self.bn1    = nn.BatchNorm2d(nfeats*8)
        #Output has the shape 8X8 with nfeats*8 channels which will be tentatively 512 channels if nfeats = 64
        self.convt3 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='zeros')
        self.bn2    = nn.BatchNorm2d(nfeats*8)
        #Output has shape 16X16 with nfeats*8
        self.convt4 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='zeros')
        self.bn3    = nn.BatchNorm2d(nfeats*8)
        #Output has shape 32X32 with nfeats*8 
        self.convt5 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='zeros')
        self.bn4    = nn.BatchNorm2d(nfeats*4)
        #Output has shape 64X64 with nfeats *4 (256)
        self.convt6 = nn.Conv2d(in_channels=nfeats*4,out_channels=nchannels,kernel_size=1,stride=1,padding=0,bias=False,padding_mode='zeros')
        #Output has shape 64X64 with 3 channels -- A RGB Image 
        
        #----------------------------------------------------------------------------------------
        #Defining the components for the residual blocks :) 
        #This is for 4x4 to 8x8 
        self.resconv1 = nn.ConvTranpose2d(in_channels=nfeats*2,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,padding_mode='zeros',bias=False)
        self.resbatch1 = nn.BatchNorm2d(nfeats*8)
        #This is for 4x4 to 16x16 via 8x8 weights
        self.resconv2 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,padding_mode='zeros',bias=False)
        self.resbatch2 = nn.BatchNorm2d(nfeats*8)
        #This is for 4x4 to 32x32 va 16x16 weights
        self.resconv3 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,padding_mode='zeros',bias=False)
        self.resbatch3 = nn.BatchNorm2d(nfeats*8)
        #------------------------------Now designing the Resblocks from 8x8 to 32x32,64x64
        self.resconv4 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,padding_mode='zeros',bias=False)
        self.resbatch4 = nn.BatchNorm2d(nfeats*8)
        self.resconv5 = nn.ConvTranspose2d(in_channels=nfeats*8,out_channels=nfeats*8,kernel_size=4,stride=2,padding=1,padding_mode='zeros',bias=False)
        self.resbatch5 = nn.BatchNorm2d(nfeats*8)
        #--------------------------------Now designing the channel attention and spatial attention-----------------
        #What has to be done here , tabling this for later 
        #Flatten the outputs , apply a fully dense ,apply softmax reshape it to an image and then mutiply it with everything
        #Mean pool , get the softmax output per channel and multiply it cast it to the image 
        #Follow C-S, C is mean pooling while S is flatten 
        #This is the last module of your DCGAN :) 
    def forward(self,x):
       
        return x
        
print("Generator model has been declared")
class Discriminator(nn.Module):
    def __init__(self,nfeats,nchannels):
        super(Discriminator,self).__init__()
        #Describing the Discriminator 
        #It is a convolutional network that goes onto flatten use Binary Cross Entropy as a loss function / softmax
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats)
        # State size is channels = nfeats , dimensions 64X64
        self.conv2 = nn.Conv2d(in_channels=nfeats, out_channels=nfeats*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats*2)
        #A State size is channels = nfeats *2 , dimensions = 32X32
        self.conv3 = nn.Conv2d(in_channels=nfeats*2, out_channels=nfeats*4, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats*4)
        #A State size is channels = nfeats *4 , dimensions =8X8
        self.conv4 = nn.Conv2d(in_channels=nfeats*4, out_channels=nfeats*8, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats*8)
         #A State size is channels = nfeats *8 , dimensions =3X3
        self.conv5 = nn.Conv2d(in_channels=nfeats*8, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        #Finished with the  definining the Discriminator Model
    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))
        return x.view(-1,1)
    
print("Discriminator model has been declared")
def PseudoHuberLoss(input,target,delta):
    if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    a=target-input
    loss = (delta**2)*((1+(a/delta)**2)**0.5-1)
    return loss 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
nfeat=64
nchannels=3
G = Generator(nfeat,nchannels).to(device)
D = Discriminator(nfeat,nchannels).to(device)
print(G)
print(D)
loss = nn.BCELoss()
L1_Loss = nn.L1Loss()
print(summary(G,input_size=(1,1,1)))
print(summary(D,input_size=(nchannels,64,64)))
#fixed_noise = torch.randn(64,64,nz,device=device)
epochs = 10000
beta1 = 0.5
beta2 = 0.999
lr = 1e-3
Goptimizer = optim.Adam(G.parameters(),lr=lr,betas=(beta1,beta2))
Doptimizer = optim.Adam(D.parameters(),lr=lr,betas=(beta1,beta2))
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(Goptimizer,mode='min',factor=0.1,patience=int(epochs),verbose=True)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(Doptimizer,mode='min',factor=0.1,patience=int(epochs),verbose=True)
true = 0.9
false = 0.1
batch_size = train_loader.batch_size
for epoch in range(epochs):
    step = 0 
    for ii,(real_images,train_labels) in enumerate(train_loader):

        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        #Training with true data
        D.zero_grad()
        labels =torch.full((batch_size,1),true,device=device)
        output = D(real_images)
        errorD_true = loss(output,labels)*0.5
        errorD_true.backward()
        D_x = output.mean().item()
        #Training with noise
        noise = torch.randn(batch_size,1,1,1,device=device)
        false_output =  G(noise)
        labels.fill_(false)
        output = D(false_output.detach())
        errorD_false = loss(output,labels)*0.5
        errorD_false.backward()
        D_G = output.mean().item()
        errorD = errorD_true+errorD_false
        Doptimizer.step()
        
        G.zero_grad()
        labels.fill_(true)
        output = D(false_output)
        errorG = loss(output,labels)
        errorG.backward(retain_graph=True)
        D_G_1 = output.mean().item()
        #Adding extra losses to the generator
        #Adding extra losses to increase the quality of images generated 
        #Extra loss functions used 
        #L1 loss 
        errorG_L1Loss =0.1*L1_Loss(false_output,real_images)
        errorG_L1Loss.backward()
        Goptimizer.step()
         #Pseudo Huber loss /Charbonnier Loss function which has been implemented by me as Pytorch doesn't have 
        #a ready made function 
        #errorG_Closs = PseudoHuberLoss(false_output.detach(),real_images,delta=0.01)
        #errorG_Closs.backward()
        
   

        if step % len(train_loader)== 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Fake/True'
                  % (epoch + 1, epochs, ii, len(train_loader),
                     errorD.item(), errorG.item(), D_x, D_G, D_G_1))
        step = step+1
    scheduler1.step(errorG)
    scheduler2.step(errorD)
   if not os.path.exists("../output_images"):
    os.mkdir("../output_images")
G.eval()
batch=25
total_images =10**4
for i in range(0,total_images,batch):
    generator = torch.randn(batch,nz,64,64,device=device)
    g_images = G(generator)
    for j in range(g_images.size(0)):
        save_image(g_images[j,:,:,:]+1./2.,os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
    import shutil 
shutil.make_archive('images','zip','../output_images)
