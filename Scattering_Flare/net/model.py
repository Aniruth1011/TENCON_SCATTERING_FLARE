import torch
import torch.nn as nn 
from .block import DoubleConv , Upsample ,  TransformerBlock  , LensComponent , FusionModule , TransformerDownsample

class Lensformer(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):

        super(Lensformer, self).__init__()

        self.conv1 = DoubleConv(input_channels , 64)

        self.transformer1 = TransformerBlock(64 , 64)

        self.enc1 = TransformerDownsample(64 , 128)

        self.enc2 = TransformerDownsample(128 , 256)

        self.enc3 = TransformerDownsample(256 , 512)

        self.lens3 = LensComponent(512)
        self.fusion3 = FusionModule(512)
        self.dec3_up = Upsample(512 , 256)
        self.dec_trans_3 = TransformerBlock(256 + 256)
        self.reduce_num_channels_3 = nn.Conv2d(512 , 256 , kernel_size=1, stride=1)

        self.lens2 = LensComponent(256)
        self.fusion2 = FusionModule(256)
        self.dec2_up = Upsample(256 , 128)
        self.dec_trans_2 = TransformerBlock(128 + 128)
        self.reduce_num_channels_2 = nn.Conv2d(256 , 128 , kernel_size=1, stride=1)

        self.lens1 = LensComponent(128)
        self.fusion1 = FusionModule(128)
        self.dec1_up = Upsample(128 , 64)
        self.dec_trans_1 = TransformerBlock(64 + 64)
        self.reduce_num_channels_1 = nn.Conv2d(128 , 64 , kernel_size=1, stride=1 )

        self.final_conv = DoubleConv(64 , output_channels)


    def forward(self , x):

        #X : (8 , 3 , 256 , 256 )
        x_conv_1 = self.conv1(x) # (8 , 64 , 256 , 256)

        x_trans_1 = self.transformer1(x_conv_1) #(8 , 64 , 256 , 256)

        x_enc_1 = self.enc1(x_trans_1) # (8 , 128 , 128 , 128 )
        x_enc_2 = self.enc2(x_enc_1) # (8 , 256 , 64 , 64)
        x_enc_3 = self.enc3(x_enc_2) # (8 , 512 , 32 , 32)

        x_lens_3 = self.lens3(x_enc_3) #(8 , 512 , 32 , 32)
        x_dec_3_input = self.fusion3(x_enc_3 , x_lens_3) # (8 , 512 , 32 , 32)
        x_dec_3_up = self.dec3_up(x_dec_3_input) # (8 , 256 , 64 , 64)
        x_dec_3 = self.dec_trans_3(torch.cat((x_dec_3_up , x_enc_2) , axis = 1)) #(8  , 512 , 64 , 64)
        x_dec_red_chan_3 = self.reduce_num_channels_3(x_dec_3) #  (8 , 256 , 64 , 64) 

        x_lens_2 = self.lens2(x_dec_red_chan_3) # (8 , 256 , 64 , 64)
        x_dec_2_input = self.fusion2(x_dec_red_chan_3 , x_lens_2) # (8 , 256 , 64 , 64)
        x_dec_2_up = self.dec2_up(x_dec_2_input) # (8 , 128 , 128 , 128)
        x_dec_2 = self.dec_trans_2(torch.cat((x_dec_2_up , x_enc_1) , axis = 1)) # (8 , 256 , 128 , 128)
        x_dec_red_chan_2 = self.reduce_num_channels_2(x_dec_2) #(8 , 128 , 128 , 128)

        x_lens_1 = self.lens1(x_dec_red_chan_2) # (8 , 128 , 128 , 128)
        x_dec_1_input = self.fusion1(x_dec_red_chan_2 , x_lens_1) # (8 , 128 , 128 , 128)
        x_dec_1_up = self.dec1_up(x_dec_1_input)  #(8 , 64 , 256 , 256)
        x_dec_1 = self.dec_trans_1(torch.cat((x_dec_1_up , x_trans_1) , axis = 1)) # (8 , 128 , 256 , 256)
        x_dec_red_chan_1 = self.reduce_num_channels_1(x_dec_1) #(8 , 64 , 256 , 256)

        x_dec_final  = self.final_conv(x_dec_red_chan_1) # (8 , 3, 256 , 256)

        return x_dec_final + x 









