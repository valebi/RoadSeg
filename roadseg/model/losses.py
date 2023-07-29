import logging
import os
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import segmentation_models_pytorch.losses as smp_l
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PatchGANDiscriminator(nn.Module):
    # initializers
    def __init__(self, in_channels, d=64, init_weights=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        self.weight_init(mean=0.0, std=0.02)

        ## Not tested
        if init_weights is not None:
            state_dict = torch.load(init_weights, map_location="cpu")
            filtered_keys = []
            for k, v in state_dict.items():
                filtered_keys.append((k.replace("module.", ""), v))
            state_dict = OrderedDict(filtered_keys)
            self.load_state_dict(state_dict)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        # x = F.sigmoid(self.conv5(x))

        return x

class PatchGANDiscriminatorLoss(nn.Module):

    def __init__(self, discriminator_lr, device='cpu' ,  discriminator_init_weights = ""):
        super().__init__()

        device  = "cpu" if device is None else device

        self._discriminator_save_path = "discriminator.pth"
        discriminator_init_weights = "discriminator.pth" if os.path.exists("discriminator.pth") else None

        self.discriminator = nn.DataParallel(PatchGANDiscriminator(in_channels=1, d = 64, init_weights = discriminator_init_weights).to(device))
        self.discriminator_criterion = nn.BCEWithLogitsLoss() ##Using pure BCE with sigmoid throws exception at autocast
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, weight_decay=1e-5)
        self._warmup_iters = 100

        #@TODO: We may add scheduler to the discriminator too
        self.wmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer,start_factor= 0.01, end_factor=1.0, total_iters= self._warmup_iters, verbose=False)
        self.actual_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30000, eta_min=1e-5, last_epoch=-1, verbose=False)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "max", factor = 0.75 ,patience=50, verbose=False)
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, [self.wmup_scheduler,  self.actual_scheduler] ,milestones=[self._warmup_iters], verbose=False)   

        self.smooth_factor = 0.1
        self.loss = 0

        self.saving_freq = 2000
        self.logging_freq = 10
        self.iter = 0
        self.history  = defaultdict(list)

    def forward(self, input, label):
        
        #We apply softmax to the predictions and smmothing to the labels before passing them to the discriminator
        input = input.log_softmax(dim=1).exp()[:,1].unsqueeze(1)
        label = label.unsqueeze(1)
        label = (1 - label) * self.smooth_factor + label * (1 - self.smooth_factor)

        if input.requires_grad:
            ##First Takes a gradient_step for the discriminator
            self.optimizer.zero_grad()

            real_pred = self.discriminator(label)
            fake_pred = self.discriminator(input.detach())

            
            real_loss = self.discriminator_criterion(real_pred, torch.ones_like (real_pred, device = label.device, requires_grad=False))
            fake_loss = self.discriminator_criterion(fake_pred, torch.zeros_like(fake_pred, device = fake_pred.device, requires_grad=False))
            


            self.loss = (real_loss + fake_loss) * 0.5
            
            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()
    
            self.iter = (self.iter + 1) % self.saving_freq
            if self.iter == 0:
                torch.save(self.discriminator.state_dict(), self._discriminator_save_path)
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.set_title("Pretraining: ")   
                ax.plot(self.history["loss"])
                ax.legend(["loss"])
                fig.savefig(f"discriminator_loss.png")
                plt.close("all")
                self.history["loss"] = []
            
            if self.iter % self.logging_freq == 0:
                self.history["loss"].append(self.loss.item())
                logging.info(f"Discriminator Loss:({self.loss:0.4f}")
                
        
        fake_pred = self.discriminator(input)
        ##Then return the discriminator loss
        return self.discriminator_criterion(fake_pred, torch.ones_like(fake_pred, device = label.device, requires_grad=False))
    

class DiceDisc(nn.Module):

    def __init__(self,discriminator_lr, device='cpu' ,  discriminator_init_weights = ""):
        super().__init__()
        self.disc = PatchGANDiscriminatorLoss(discriminator_lr, device, discriminator_init_weights)
        self.dice = smp_l.DiceLoss(mode="multiclass")

        self.disc_weight = 0.01
        self.dice_weight = 1.0

    def forward(self, input, label):
        return self.disc_weight * self.disc(input, label) + self.dice_weight * self.dice(input, label)
    
        
