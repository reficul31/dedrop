import os
import torch
import numpy as np

class Trainer:
    def __init__(self, criterion, optimizer, scheduler, data_loader, root_dir, batch_size):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.batch_size = batch_size

    def train_model(self, name, model, epochs=100, checkpoint_epoch = 0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if checkpoint_epoch != 0:
            print("Continuing training from epoch = {}".format(checkpoint_epoch))
        
        train_loss = []
        for epoch in range(checkpoint_epoch, epochs):
            model.train()
            batch_step_size = len(self.data_loader.dataset) / self.batch_size
            
            log_loss = []
            for batch_idx, sample in enumerate(self.data_loader):
                rain, clean = sample
                rain = rain.to(device).float()
                clean = clean.to(device).float()
                outputs, _ = model(rain)
                pixel_metric = self.criterion(outputs, clean)
                loss = -pixel_metric

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                log_loss.append(pixel_metric.item())
                if batch_idx % 25 == 0:
                    print("Epoch {} : {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch, 'Train', batch_idx, int(batch_step_size), loss.item()))
                train_loss.append(np.mean(log_loss))

            if not os.path.isdir(os.path.join(self.root_dir, name)):
                os.makedirs(os.path.join(self.root_dir, name))

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, os.path.join(self.root_dir, name, "checkpoint_{}.tar".format(epoch)))
                np.save(os.path.join(self.root_dir, name, "train-loss-epoch-{}.npy".format(epoch)), train_loss)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, os.path.join(self.root_dir, name, "checkpoint_latest.tar"))
        return model, train_loss