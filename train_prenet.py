import torch

from ssim import SSIM
from models import PReNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import get_dataset
from train import Trainer

if __name__ == '__main__':
    name = 'prenet'
    root_dir = "/home/sb4539/dedrop"
    
    epochs, batch_size = 100, 18

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PReNet().to(device)
    
    train_dataset = get_dataset(phase='train')
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = SSIM()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80])

    trainer = Trainer(criterion, optimizer, scheduler, dataloader, root_dir, batch_size)
    model, _ = trainer.train_model(name, model, epochs=epochs)
