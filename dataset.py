import os
import cv2
import h5py
import torch
import numpy as np
import random as rd

from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, phase):
        super().__init__()
        
        self.size = (704, 448)
        file_path = os.path.dirname(os.path.realpath(__file__))

        self.rain_folder_path = os.path.join(file_path, phase, "data")
        self.clean_folder_path = os.path.join(file_path, phase, "gt")

        self.rain_image_list = os.listdir(self.rain_folder_path)
        self.clean_image_list = os.listdir(self.clean_folder_path)

        self.rain_image_list.sort()
        self.clean_image_list.sort()

        assert len(self.rain_image_list) == len(self.clean_image_list)   
        
    def __len__(self):
        return len(self.rain_image_list)
    
    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_folder_path, self.clean_image_list[idx])
        rain_path = os.path.join(self.rain_folder_path, self.rain_image_list[idx])
        
        clean = cv2.imread(clean_path)
        clean = cv2.resize(clean, self.size)
        b, g, r = cv2.split(clean)
        clean = cv2.merge([r, g, b])

        rain = cv2.imread(rain_path)
        rain = cv2.resize(rain, self.size)
        b, g, r = cv2.split(rain)
        rain = cv2.merge([r, g, b])

        clean = np.float32(clean) / 255
        rain = np.float32(rain) / 255

        clean = clean.transpose(2, 0, 1)
        rain = rain.transpose(2, 0, 1)
        
        return torch.Tensor(rain), torch.Tensor(clean)

class TrainDataset(Dataset):
    def __init__(self, phase):
        super(TrainDataset, self).__init__()

        file_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(file_path, phase)

        rain_path = os.path.join(self.data_dir, 'train_rain.h5')
        clean_path = os.path.join(self.data_dir, 'train_clean.h5')

        rain_data = h5py.File(rain_path, 'r')
        clean_data = h5py.File(clean_path, 'r')

        self.keys = list(rain_data.keys())
        rd.shuffle(self.keys)
        
        rain_data.close()
        clean_data.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        rain_path = os.path.join(self.data_dir, 'train_rain.h5')
        clean_path = os.path.join(self.data_dir, 'train_clean.h5')

        rain_data = h5py.File(rain_path, 'r')
        clean_data = h5py.File(clean_path, 'r')

        key = self.keys[index]
        rain = np.array(rain_data[key])
        clean = np.array(clean_data[key])

        rain_data.close()
        clean_data.close()

        return torch.Tensor(rain), torch.Tensor(clean)

def get_dataset(phase):
    if phase == 'train':
        return TrainDataset(phase=phase)
    else:
        return TestDataset(phase=phase)
