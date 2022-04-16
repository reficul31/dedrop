import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from math import log
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import get_dataset
from models import VAE, InpaintNet, Discriminator

name = 'dedrop_gan'
root_dir = "/home/sb4539/dedrop"

epochs, batch_size = 100, 32
print_frequency, save_checkpoint_frequency = 500, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netDropGen = VAE().to(device)
netInpaint = InpaintNet().to(device)
netDisc = Discriminator().to(device)

optDisc = Adam(netDisc.parameters(), lr=1e-3)
optDropGen = Adam(netDropGen.parameters(), lr=1e-3)
optInpaint = Adam(netInpaint.parameters(), lr=1e-3)

schedulerDisc = MultiStepLR(optDisc, [50, 80, 95], gamma=0.1)
schedulerDropGen = MultiStepLR(optDropGen, [50, 80, 95], gamma=0.1)
schedulerInpaint = MultiStepLR(optInpaint, [50, 80, 95], gamma=0.1)

checkpoint_epoch = 0
if not os.path.isdir(os.path.join(root_dir, name)):
    checkpoint_epoch = 0
    print("No checkpoint folder found. Starting from epoch 0.")
    os.makedirs(os.path.join(root_dir, name))
else:
    checkpoint_drop_gen = torch.load(os.path.join(root_dir, name, "checkpoint_drop_gen_latest.tar"))
    checkpoint_epoch = int(checkpoint_drop_gen['epoch'])
    netDropGen.load_state_dict(checkpoint_drop_gen['model_state_dict'])
    optDropGen.load_state_dict(checkpoint_drop_gen['optimizer_state_dict'])
    schedulerDropGen.load_state_dict(checkpoint_drop_gen['scheduler_state_dict'])

    checkpoint_disc = torch.load(os.path.join(root_dir, name, "checkpoint_disc_latest.tar"))
    assert checkpoint_epoch == int(checkpoint_disc['epoch'])
    netDisc.load_state_dict(checkpoint_disc['model_state_dict'])
    optDisc.load_state_dict(checkpoint_disc['optimizer_state_dict'])
    schedulerDisc.load_state_dict(checkpoint_disc['scheduler_state_dict'])

    checkpoint_inpaint = torch.load(os.path.join(root_dir, name, "checkpoint_inpaint_latest.tar"))
    assert checkpoint_epoch == int(checkpoint_inpaint['epoch'])
    netInpaint.load_state_dict(checkpoint_inpaint['model_state_dict'])
    optInpaint.load_state_dict(checkpoint_inpaint['optimizer_state_dict'])
    schedulerInpaint.load_state_dict(checkpoint_inpaint['scheduler_state_dict'])

train_dataset = get_dataset(phase='train')
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

netDropGen.train()
netInpaint.train()
netDisc.train()

train_loss = []
for epoch in range(checkpoint_epoch, epochs):
    batch_step_size = len(dataloader.dataset) / batch_size
    
    log_loss = []
    start = time.time()
    for batch_idx, sample in enumerate(dataloader):
        rain, clean = sample
        rain = rain.to(device).float()
        clean = clean.to(device).float()

        disc_out_real, _ = netDisc(clean)
        disc_loss_real = -torch.mean(disc_out_real)

        mask, mu, logvar, _ = netDropGen(rain)
        _, _, clean_fake = netInpaint(rain, mask)

        logvar.clamp(min=log(1e-8), max=log(1e4))
        var = torch.exp(logvar)
        kl_gauss = torch.mean(mu ** 2 + (var - 1 - logvar)) / 2

        disc_out_fake, _ = netDisc(clean_fake.detach())
        disc_loss_fake = torch.mean(disc_out_fake)
        
        alpha = torch.rand(clean.size(0), 1, 1, 1).cuda().expand_as(clean)
        interpolated = Variable(alpha * clean.data + (1 - alpha) * clean_fake.data, requires_grad=True)
        
        out, _, _ = netDisc(interpolated)
        grad = torch.autograd.grad(outputs=out,
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        disc_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        with torch.set_grad_enabled(True):
            netDisc.zero_grad()
            
            lossDisc = disc_loss_real + disc_loss_fake + 10 * disc_loss_gp
            lossDisc.backward()
            
            optDisc.step()
            if (batch_idx + 1) % 5 == 0:
                netDropGen.zero_grad()
                netInpaint.zero_grad()
                
                gen_out_fake, _ = netDisc(clean_fake.detach())
                gen_loss_fake = -torch.mean(gen_out_fake)
                lossGen = gen_loss_fake + kl_gauss

                lossGen.backward()
                optDropGen.step()
                optInpaint.step()
            schedulerDisc.step()
            schedulerDropGen.step()
            schedulerInpaint.step()
        
        loss = F.mse_loss(clean, clean_fake)
        log_loss.append(loss.item())
        if batch_idx % print_frequency == 0:
            print("Epoch {} : {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch + 1, 'Train', batch_idx, int(batch_step_size), loss.item()))
    
    train_loss.append(np.mean(log_loss))
    print("Epoch {} done: Time = {}, Mean Loss = {}".format(epoch + 1, time.time() - start, train_loss[-1]))

    if epoch % save_checkpoint_frequency == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': netDropGen.state_dict(),
            'optimizer_state_dict': optDropGen.state_dict(),
            'scheduler_state_dict': schedulerDropGen.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_drop_gen_{}.tar".format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': netDisc.state_dict(),
            'optimizer_state_dict': optDisc.state_dict(),
            'scheduler_state_dict': schedulerDisc.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_disc_{}.tar".format(epoch)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': netInpaint.state_dict(),
            'optimizer_state_dict': optInpaint.state_dict(),
            'scheduler_state_dict': schedulerInpaint.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_inpaint_{}.tar".format(epoch)))
        np.save(os.path.join(root_dir, name, "train-loss-epoch-{}.npy".format(epoch)), train_loss)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': netDropGen.state_dict(),
            'optimizer_state_dict': optDropGen.state_dict(),
            'scheduler_state_dict': schedulerDropGen.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_drop_gen_latest.tar"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': netDisc.state_dict(),
            'optimizer_state_dict': optDisc.state_dict(),
            'scheduler_state_dict': schedulerDisc.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_disc_latest.tar"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': netInpaint.state_dict(),
            'optimizer_state_dict': optInpaint.state_dict(),
            'scheduler_state_dict': schedulerInpaint.state_dict()
        }, os.path.join(root_dir, name, "checkpoint_inpaint_latest.tar"))
