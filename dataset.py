import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

class DedropDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None, normalize = True, size=(480, 760), window=128):
        super().__init__()
        
        self.size = size
        self.phase = phase
        self.transform = transform
        self.window = window
        self.normalize = normalize

        self.rain_folder_path = os.path.join(root_dir, phase, "data")
        self.clean_folder_path = os.path.join(root_dir, phase, "gt")

        self.rain_image_list = os.listdir(self.rain_folder_path)
        self.clean_image_list = os.listdir(self.clean_folder_path)

        self.rain_image_list.sort()
        self.clean_image_list.sort()

        assert len(self.rain_image_list) == len(self.clean_image_list)   
        
    def __len__(self):
        return len(self.rain_image_list)
    
    def get_random_patches(self, rain, clean):
        endh, endw = self.size
        width, height = np.random.randint(0, endw - self.window-1), np.random.randint(0, endh - self.window-1)

        rain_patch = rain[width: width+self.window, height: height+self.window, :]
        clean_patch = clean[width: width+self.window, height: height+self.window, :]
        return rain_patch, clean_patch
    
    def __getitem__(self, idx):
        clean_image_path = os.path.join(self.clean_folder_path, self.clean_image_list[idx])
        rain_image_path = os.path.join(self.rain_folder_path, self.rain_image_list[idx])
        
        rain = Image.open(rain_image_path).resize(self.size).convert("RGB")
        clean = Image.open(clean_image_path).resize(self.size).convert("RGB")

        rain, clean = self.get_random_patches(np.array(rain), np.array(clean))
        
        if self.normalize:
            rain, clean = rain / 255, clean / 255
        
        if self.transform is None:
            return rain, clean

        return self.transform(rain), self.transform(clean)
        

def get_dataset(root_dir, phase):
    return DedropDataset(
        root_dir, 
        phase=phase, 
        transform=Compose([ToTensor()])
    )
