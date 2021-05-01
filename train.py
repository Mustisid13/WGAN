import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator,initialize_weight
import time
from gp import gradient_penalty 
#Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#learning rate
lr = 1e-4
#batch size
bs = 64
image_size = 64
channels_img = 3
z_dim = 100
num_epoch = 5
features_disc = 64
features_gen = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
# setting transform for dataset
transform = transforms.Compose(
    [ transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
     )]
)
#loading dataset
dataset = datasets.ImageFolder(root="celeb_dataset",transform=transform)
loader = DataLoader(dataset,batch_size = bs,shuffle = True)

#initialize models
gen = Generator(z_dim,channels_img,features_gen).to(device)
critic = Discriminator(channels_img,features_disc).to(device)
#initialize weights of model
initialize_weight(gen)
initialize_weight(critic)

#initialize optimizer
opt_gen = optim.Adam(gen.parameters(),lr=lr,betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(),lr=lr,betas=(0.0,0.9))
#initialize loss fuction
# criterion = nn.BCELoss()

#generatiing fixed noise for better visualzization
fixed_noise = torch.randn(32,z_dim,1,1).to(device)

#writer for tensorboard
writer_real = SummaryWriter(f'logs/real')
writer_fakes = SummaryWriter(f'logs/fake')
step = 0

gen.train()
critic.train()

#training DCGAN
for epoch in range(num_epoch):
    for batch_idx,(real,_) in enumerate(loader):
        start = time.time()
        real = real.to(device)
        
        ##train discriminator
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((bs,z_dim,1,1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            
            gp = gradient_penalty(critic,real,fake,device=device)
            # print(gp)
            loss_critic = (-(torch.mean(critic_real)-torch.mean(critic_fake)) + LAMBDA_GP * gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
                
        #train gen
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}/{num_epoch} Batch: {batch_idx} Device: {device} Time: {(time.time() - start)}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                
                img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                
                writer_real.add_image("Real",img_grid_real,global_step = step)
                writer_fakes.add_image("Fake",img_grid_fake,global_step = step)
                
            step+=1
        
        