from torch.utils.data import Dataset
import os 
import cv2 as cv
from torchvision.transforms import ToTensor , Resize , Compose
 
class ScatteringFlareDataset(Dataset):

    def __init__(self , data_path , image_size):

        self.data_dir = data_path

        self.compound_flare_image_paths = [os.path.join(self.data_dir , 'Compound_Flare' , img) for img in os.listdir(os.path.join(self.data_dir , 'Compound_Flare'))]
        self.core_image_paths = [os.path.join(self.data_dir , 'Core' , img) for img in os.listdir(os.path.join(self.data_dir , 'Core'))]
        self.glare_with_shimmer_image_paths = [os.path.join(self.data_dir , 'Glare_with_shimmer' , img) for img in os.listdir(os.path.join(self.data_dir , 'Glare_with_shimmer'))]
        self.streak_image_paths = [os.path.join(self.data_dir , 'Streak' , img) for img in os.listdir(os.path.join(self.data_dir , 'Streak'))]

        self.total_inputs = self.compound_flare_image_paths + self.core_image_paths + self.glare_with_shimmer_image_paths +  self.streak_image_paths

        self.output_image_paths = [os.path.join(self.data_dir , 'Light_Source' , img) for img in os.listdir(os.path.join(self.data_dir , 'Light_Source'))]

        self.transform = Compose([ToTensor() , Resize(image_size)])

    def __len__(self):

        return 2  #len(self.total_inputs)
    
    def __getitem__(self, idx):

        input_img_path = self.total_inputs[idx]
        output_img_path = self.output_image_paths[idx%len(self.output_image_paths)]

        input_img = cv.imread(input_img_path)
        output_img = cv.imread(output_img_path)

        input_img_transform = self.transform(input_img)
        output_img_transform = self.transform(output_img)

        return input_img_transform , output_img_transform





