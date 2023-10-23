import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import spectral_norm


# Configuration
config = {
    "dataroot": "",
    "workers": 0,
    "batch_size": 8,
    "image_size": 256,
    "lr": 0.01,
    "lrD": 0.01,
    "num_epochs": 1001,
    "start_epoch": 1,
    "nc": 3,
    "nz": 512
}

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class CustomActivation(nn.Module):
    def __init__(self, in_channels):
        super(CustomActivation, self).__init__()

        # Learnable parameters
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))
        self.beta = nn.Parameter(torch.FloatTensor([0.0]))
        self.theta = nn.Parameter(torch.FloatTensor([1.0]))
        self.radius = nn.Parameter(torch.FloatTensor([0.5]))
        self.gamma = nn.Parameter(torch.FloatTensor([0.1]))
        self.delta = nn.Parameter(torch.FloatTensor([0.0]))

        # Batch Normalization
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Complex representation
        z = torch.complex(x, x)

        # Euler's rotation with a twist: added gamma and delta for more flexibility
        euler_rotation = torch.exp((self.theta + self.gamma) * 1j * self.delta) * z
        euler_term = self.radius * (euler_rotation.real + euler_rotation.imag)
        
        # For x > 0: pos = euler_term + alpha
        pos = euler_term + self.alpha
        
        # For x <= 0: neg = euler_term - beta
        neg = euler_term - self.beta

        # Combine both sides
        act = torch.where(x > 0, pos, neg)
        
        # Apply Batch Normalization
        activated = self.batch_norm(act)

        return activated
    
class CustomActivationD(nn.Module):
    def __init__(self):
        super(CustomActivationD, self).__init__()

        # Learnable parameters for Euler's rotation
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))
        self.beta = nn.Parameter(torch.FloatTensor([0.0]))
        self.theta = nn.Parameter(torch.FloatTensor([1.0]))
        self.radius = nn.Parameter(torch.FloatTensor([0.5]))
        self.gamma = nn.Parameter(torch.FloatTensor([0.1]))
        self.delta = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x):
        # Complex representation
        z = torch.complex(x, x)

        # Euler's rotation with a twist: added gamma and delta for more flexibility
        euler_rotation = torch.exp((self.theta + self.gamma) * 1j * self.delta) * z
        euler_term = self.radius * (euler_rotation.real + euler_rotation.imag)
        
        # For x > 0: pos = euler_term + alpha
        pos = euler_term + self.alpha
        
        # For x <= 0: neg = euler_term - beta
        neg = euler_term - self.beta

        # Combine both sides
        act = torch.where(x > 0, pos, neg)

        return act
    
class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpBlock, self).__init__()

        self.drop_val = nn.Parameter(torch.FloatTensor([0.0]))
        red_channels = in_channels // 2
        out_channels = in_channels // 4

        self.skip_val = nn.Parameter(torch.FloatTensor([0.0]))

        self.k1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(in_channels),

            nn.ConvTranspose2d(in_channels, red_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, red_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(out_channels),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(out_channels)

        )

        self.k2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            CustomActivation(in_channels),

            nn.ConvTranspose2d(in_channels, red_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, red_channels, kernel_size=3, stride=1, padding=1, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(out_channels),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            CustomActivation(out_channels)
        )

        self.k3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False),
            CustomActivation(in_channels),

            nn.ConvTranspose2d(in_channels, red_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, red_channels, kernel_size=5, stride=1, padding=2, bias=False),
            CustomActivation(red_channels),

            nn.ConvTranspose2d(red_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            CustomActivation(out_channels),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            CustomActivation(out_channels)
        )

        self.channel_correct = nn.Sequential(
            nn.ConvTranspose2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(out_channels),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            CustomActivation(out_channels),
        )

        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(out_channels),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            CustomActivation(out_channels),
        )

    def forward(self, x):
        drop = self.drop_val.item()
        k1 = self.k1(x)
        k2 = self.k2(x)  
        k3 = self.k3(x)
        cat = torch.cat((k1, k2, k3), dim=1)
        cor = self.channel_correct(cat)
        skip = self.skip(x) * self.skip_val
        out = cor + skip
        final = F.dropout2d(out, p=drop, training=self.training)
        return final


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownBlock, self).__init__()

        self.drop_val = nn.Parameter(torch.FloatTensor([0.0]))
        inc_channels = in_channels * 2
        out_channels = in_channels * 4

        self.skip_val = nn.Parameter(torch.FloatTensor([0.0]))

        self.k1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(in_channels, inc_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, inc_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            CustomActivationD()

        )

        self.k2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(in_channels, inc_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, inc_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            CustomActivationD()
        )

        self.k3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(in_channels, inc_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, inc_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(inc_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            CustomActivationD()
        )

        self.channel_correct = nn.Sequential(
            spectral_norm(nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            CustomActivationD(),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            CustomActivationD()
        )

        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            CustomActivationD(),

            nn.AvgPool2d(4),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            CustomActivationD()
        )

    def forward(self, x):
        drop = self.drop_val.item()
        k1 = self.k1(x)
        k2 = self.k2(x)  
        k3 = self.k3(x)
        cat = torch.cat((k1, k2, k3), dim=1)
        cor = self.channel_correct(cat)
        skip = self.skip(x) * self.skip_val
        out = cor + skip
        final = F.dropout2d(out, p=drop, training=self.training)
        return final
    
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        self.Initial = nn.Sequential(
            nn.ConvTranspose2d(nz, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            CustomActivation(1024),
        )

        self.upleg1 = nn.Sequential(
            UpBlock(1024), 
            UpBlock(256), 
            UpBlock(64), 
            UpBlock(16)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(4, 3, kernel_size=5, stride=1, padding=2, bias=False)
        )

    def forward(self, x):
        init = self.Initial(x)
        
        up = self.upleg1(init) 

        out = self.final(up)
        
        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Initial = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2, bias=False)),
            CustomActivationD()
        )

        self.down = nn.Sequential(
        DownBlock(4),  
        DownBlock(16), 
        DownBlock(64),
        DownBlock(256)
        )

        self.final = nn.Sequential(        
            spectral_norm(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=False))
        )

    def forward(self, x):
        init = self.Initial(x)

        down = self.down(init)

        out = self.final(down)
        
        return out.squeeze()
    
Generator, Discriminator

G = Generator(config["nz"]).to(device)
D = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        std_dev = 1.0 / (m.in_channels * m.kernel_size[0] * m.kernel_size[1]) ** 0.5
        torch.nn.init.normal_(m.weight, mean=0., std=std_dev)
        
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif hasattr(m, 'weight') and classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


G.apply(weights_init)
D.apply(weights_init)

def train_gan(netG, netD, dataloader, config, device):
    optimizerD = optim.RMSprop(netD.parameters(), lr=config["lrD"])
    optimizerG = optim.RMSprop(netG.parameters(), lr=config["lr"])
    CHECKPOINT_SAVE_INTERVAL = 100

    for epoch in range(config["start_epoch"], config["num_epochs"]):
        for batch_idx, (real_images, _) in enumerate(dataloader):

            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            noise = torch.randn(batch_size, config["nz"], 1, 1, device=device)

            # Discriminator Update
            netD.zero_grad()
            logits_real = netD(real_images)
            fake_images = netG(noise).detach()
            logits_fake = netD(fake_images)

            # Wasserstein loss for Discriminator
            lossD = -torch.mean(logits_real) + torch.mean(logits_fake)
            lossD.backward()
            optimizerD.step()

            # Generator Update
            netG.zero_grad()
            fake_images = netG(noise)
            logits_fake_updated = netD(fake_images)

            # Wasserstein loss for Generator
            lossG = -torch.mean(logits_fake_updated)
            lossG.backward()
            optimizerG.step()

            checkpoints = [len(dataloader) // 3, 2 * len(dataloader) // 3, len(dataloader) - 1]
            if batch_idx in checkpoints:
                fraction = checkpoints.index(batch_idx) + 1
                print(f"[{epoch}/{config['num_epochs']}][{batch_idx}/{len(dataloader)}] (Epoch {fraction}/3) Loss D: {lossD.item()}, Loss G: {lossG.item()}")

                # Save the generated images grouped into one image
                save_image(fake_images, f"./trainpics/images_{epoch}_{batch_idx}_fraction_{fraction}.png", normalize=True, nrow=int(len(fake_images)**0.5))
        
        # Save model checkpoints less frequently
        if epoch % CHECKPOINT_SAVE_INTERVAL == 0 or epoch == config["num_epochs"] - 1:
            torch.save(netG.state_dict(), f'./generator/gen_{epoch}.pth')
            torch.save(netD.state_dict(), f'./discriminator/disc_{epoch}.pth')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(config["dataroot"], transform=transform)
sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    sampler=sampler, 
    num_workers=config["workers"],
    pin_memory=True,  
    drop_last=True    
)

train_gan(G, D, dataloader, config, device)