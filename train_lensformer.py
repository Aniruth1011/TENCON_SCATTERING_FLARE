import torch
import torch.nn as nn

import torch.optim as optim

import os
from tqdm import tqdm

from options import options

from Scattering_Flare.utils.create_dataloader import training_dataloader
from Scattering_Flare.net.model import Lensformer

torch.cuda.empty_cache()

def tensor_to_cv2_gray(tensor):
    tensor = tensor.permute(0, 2, 3, 1)

    r_factor = torch.tensor(0.2989, dtype=tensor.dtype )
    g_factor = torch.tensor(0.5870, dtype=tensor.dtype ) 
    b_factor = torch.tensor(0.1140, dtype=tensor.dtype ) 

    gray_tensor = tensor[..., 0] * r_factor + tensor[..., 1] * g_factor + tensor[..., 2] * b_factor

    gray_tensor = gray_tensor.unsqueeze(1).permute(0, 3, 1, 2)

    return gray_tensor

train_dataset, train_dataloader = training_dataloader(options.batch_size, options.no_of_workers, options.train_dataset_path, options.image_size)

model = Lensformer()
model = model.to(options.device)

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

        print("Lensformer Epoch Loss :" , epoch_loss)

        if (epoch % options.save_every == 0):
            torch.save(model.state_dict(), f'ckpt/lensformer/lensformer_{epoch}.pth')

    torch.save(model.state_dict(), options.lensformer_model_path)
