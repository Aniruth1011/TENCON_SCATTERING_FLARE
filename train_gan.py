import torch
import torch.nn as nn
from torch.utils.data import DataLoader 

import torch.optim as optim

import os
from tqdm import tqdm

from options import options
from utils import load_deflared_mask_dataset 

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


if options.with_prompts:
    model = Prompted_InpaintGenerator(options)
    model_org = InpaintGenerator(options)
    model_org.load_state_dict(torch.load(os.path.join(options.pretrained, 'G0000000.pt'), map_location=options.device), strict=False)
    model.load_state_dict(load_pretrained_weights(model_org, model), strict=False)
else:
    model = InpaintGenerator(options)
    model.load_state_dict(torch.load(os.path.join(options.pretrained, 'G0000000.pt'), map_location=options.device), strict=False)

model = model.to(options.device)

image_mask_dataset = load_deflared_mask_dataset()
image_mask_loader = DataLoader(image_mask_dataset , batch_size =  options.batch_size , num_workers = options.no_of_workers)

optimizer = optim.Adam(model.parameters() , lr=options.lrg)
criterion = nn.MSELoss()

if __name__ == '__main__':
    
    for epoch in tqdm(range(options.num_epochs)):

        model.train()

        running_loss = 0.0

        for batch_idx, (deflared_img , mask) in enumerate(image_mask_loader):

            deflared_img = deflared_img.to(options.device)
            mask = mask.to(options.device)

            optimizer.zero_grad()
            outputs = model(deflared_img , mask)
            loss = criterion(outputs, deflared_img)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(image_mask_dataset)

        print("GAN Epoch Loss :" , epoch_loss)

        if (epoch % options.save_every == 0):
            if options.with_prompts:
                torch.save(model.state_dict(), f'ckpt/promptgan/promptgan_{epoch}.pth')
            else:
                torch.save(model.state_dict(), f'ckpt/gan/gan_{epoch}.pth')

    if options.with_prompts:
        torch.save(model.state_dict(), options.promptgan_model_path)
    else:
        torch.save(model.state_dict(), options.gan_model_path)

