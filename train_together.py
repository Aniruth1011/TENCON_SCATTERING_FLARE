import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm

from options import options

from Scattering_Flare.utils.create_dataloader import training_dataloader
from Scattering_Flare.net.model import Lensformer

from GAN.src.model.promptgan import Prompted_InpaintGenerator 
from GAN.src.model.aotgan import InpaintGenerator


torch.cuda.empty_cache()

def load_pretrained_weights(original_model, new_model):
    orig_ = {}
    new_ = {}

    for (name, param) in original_model.named_parameters():
        orig_[name] = param 

    for (name, param) in new_model.named_parameters():
        if name in orig_:
            new_[name] = orig_[name]
        else:
            new_[name] = torch.randn_like(param) / torch.norm(torch.randn_like(param))
    return new_


def tensor_to_cv2_gray(tensor):
    tensor = tensor.permute(0, 2, 3, 1)

    r_factor = torch.tensor(0.2989, dtype=tensor.dtype )
    g_factor = torch.tensor(0.5870, dtype=tensor.dtype ) 
    b_factor = torch.tensor(0.1140, dtype=tensor.dtype ) 

    gray_tensor = tensor[..., 0] * r_factor + tensor[..., 1] * g_factor + tensor[..., 2] * b_factor

    gray_tensor = gray_tensor.unsqueeze(1).permute(0, 3, 1, 2)

    return gray_tensor

    
class Combined_Model(nn.Module):

    def __init__(self , options):
        super(Combined_Model, self).__init__()
        self.pretrained_state_dict = torch.load(os.path.join(options.pretrained, 'G0000000.pt'), map_location=torch.device('cuda'))
        self.unet = Lensformer().to(options.device)
        if options.with_prompts:
            self.inpainter = Prompted_InpaintGenerator(options)
            self.org_gan = InpaintGenerator(options)
            self.org_gan.load_state_dict(torch.load(os.path.join(options.pretrained, 'G0000000.pt'), map_location=options.device), strict=False)
            self.inpainter.load_state_dict(load_pretrained_weights(self.org_gan, self.inpainter), strict=False)
        else:
            self.inpainter = InpaintGenerator(options)
            self.inpainter.load_state_dict(torch.load(os.path.join(options.pretrained, 'G0000000.pt'), map_location=options.device), strict=False)
        self.models = [self.unet, self.inpainter]
        self.learnable_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    
    def forward(self , x):

        unet_output = self.unet(x)
        unet_output_gray = tensor_to_cv2_gray(unet_output)
        flary_img_gray = tensor_to_cv2_gray(x)
        difference = flary_img_gray - unet_output_gray
        difference_tensor = difference.clone().to(options.device)
        binary_mask = F.relu(difference_tensor - self.learnable_threshold )
        inpainted_output = self.inpainter(unet_output , binary_mask.permute(0 , 2  , 1 , 3))
        interpolated_inpainted_output = F.interpolate(inpainted_output , (options.image_size , options.image_size))

        return interpolated_inpainted_output
    
model = Combined_Model(options)
model = model.to(options.device)

train_dataset, train_dataloader = training_dataloader(options.batch_size, options.no_of_workers, options.train_dataset_path, options.image_size)

optimizer = optim.Adam(model.parameters() , lr=options.learning_rate)

criterion = nn.MSELoss()

if __name__ == '__main__':
    
    for epoch in tqdm(range(options.num_epochs)):

        model.train()

        running_loss = 0.0

        for batch_idx, (flary_img, deflared_img) in enumerate(train_dataloader):
            flary_img = flary_img.to(options.device)
            deflared_img = deflared_img.to(options.device)
            optimizer.zero_grad()
            outputs = model(flary_img)
            loss = criterion(outputs, deflared_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataset)
        print("Epoch Loss :" , epoch_loss)
        print(f"Learnable Threshold: {model.learnable_params[0].item()}")

        if (epoch % options.save_every == 0):
            torch.save(model.state_dict(), f'ckpt/unet_model_{epoch}.pth')

    torch.save(model.state_dict(), options.model_path)