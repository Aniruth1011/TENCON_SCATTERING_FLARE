import torch
from torch.utils.data import Dataset
import os 
import cv2 as cv
from torchvision.transforms import ToTensor , Resize , Compose

class load_deflared_mask_dataset(Dataset):

    def __init__(self , image_size):

        self.deflared_images = os.listdir(r'Deflared')
        self.masks = os.listdir(r'Masks')

        self.deflared_image_paths = os.listdir(self.deflared_image_paths)
        self.mask_paths = os.listdir(self.masks)

        self.transform = Compose([ToTensor() , Resize(image_size)])


    def len(self):

        return len(self.deflared_image_paths)
    
    def __getitem__(self, idx):

        deflared_path  = self.deflared_image_paths[idx]
        mask_path = self.mask_paths[idx]

        deflared_img = cv.imread(deflared_path)
        mask_img = cv.imread(mask_path)

        return self.transform(deflared_img) , self.transform(mask_img)
