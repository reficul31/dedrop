import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class DedropDataset(Dataset):
    def __init__(self, root_dir, phase = 'train', transform=None):
        super().__init__()
        
        self.phase = phase
        self.transform = transform

        self.rain_folder_path = os.path.join(root_dir, phase, "data")
        self.clean_folder_path = os.path.join(root_dir, phase, "gt")

        self.rain_image_list = os.listdir(self.rain_folder_path)
        self.clean_image_list = os.listdir(self.clean_folder_path)

        self.rain_image_list.sort()
        self.clean_image_list.sort()

        assert len(self.rain_image_list) == len(self.clean_image_list)   
        
    def __len__(self):
        return len(self.rain_image_list)
    
    def __getitem__(self, idx):
        clean_image_path = os.path.join(self.clean_folder_path, self.clean_image_list[idx])
        rain_image_path = os.path.join(self.rain_folder_path, self.rain_image_list[idx])
        
        rain = Image.open(rain_image_path).convert("RGB")
        clean = Image.open(clean_image_path).convert("RGB")

        if self.transform is None:
            return rain, clean

        return self.transform(rain), self.transform(clean)
        

def get_dataset(root_dir, phase):
    return DedropDataset(
        root_dir, 
        phase=phase, 
        transform=Compose([
            Resize((240, 360)),
            ToTensor()
        ])
    )
