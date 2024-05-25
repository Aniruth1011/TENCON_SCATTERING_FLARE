import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LensComponent(nn.Module):
    def __init__(self, input_dim, pretrained=True):
        super(LensComponent, self).__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim)) 
        nn.init.normal_(self.weight) 
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(input_dim , input_dim , kernel_size=3, padding=1 )
        self.tanh = nn.Tanh()
 
    def forward(self, x):

        conv_output_1 = self.conv1(x)

        conv_output_2 = self.conv2(conv_output_1)
        
        weighted_output = conv_output_2 * self.weight.view(1, self.input_dim, 1, 1)
        
        output = self.tanh(weighted_output)
        
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze()
        max_out = self.max_pool(x).squeeze()
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=7, stride=1, padding=3):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv0 = nn.Conv2d(in_channels * 2 , in_channels , kernel_size=kernel_size, stride=stride, padding=padding )
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        out = F.relu(self.conv0(torch.cat([avg_out, max_out], dim=1)))
        
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        return self.sigmoid(out)

class AttentionFusion(nn.Module):
    def __init__(self, in_planes):
        super(AttentionFusion, self).__init__()
        self.fc = nn.Linear(in_planes * 2, in_planes)

        
    def forward(self, channel_attention, spatial_attention):
        spatial_attention = F.interpolate(spatial_attention, size=channel_attention.size()[2:], mode='nearest')
        if spatial_attention.size() != channel_attention.size():
            spatial_attention = F.pad(spatial_attention, (0, 0, 0, 128 - spatial_attention.size(1)), mode='constant', value=0)
        result = spatial_attention * channel_attention
        return result
    

class TransformerBlock(nn.Module):
    def __init__(self, in_channels , reduction_ratio=16, kernel_size=7, stride=1, padding=3):
        super(TransformerBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Channel-wise attention
        x_ca = self.channel_attention(x) * x

        x_sa = self.spatial_attention(x_ca) * x_ca
        
        # Convolutional layers
        out = self.conv1(x_sa)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Skip connection
        out += x
        
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
 
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # print('6. Down')
        return self.maxpool_conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class FusionModule(nn.Module):
    def __init__(self, input_channels):

        super(FusionModule, self).__init__()
        self.block_weight = nn.Parameter(torch.randn(input_channels))  
        self.lens_weight = nn.Parameter(torch.randn(input_channels))  

    def forward(self, encoder_output, operation_output):

        weighted_encoder_output = encoder_output * self.block_weight.view(1, -1, 1, 1)
        weighted_operation_output = operation_output * self.lens_weight.view(1, -1, 1, 1)

        combined_output = weighted_encoder_output + weighted_operation_output
        return combined_output

class TransformerDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerDownsample, self).__init__()
        self.transformer = TransformerBlock(in_channels)
        self.down = Down(in_channels, out_channels)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.down(x)
        return x

class Upsample(nn.Module):

    def __init__(self , in_channels , out_channels):
        super(Upsample , self).__init__()
        self.up = nn.ConvTranspose2d(in_channels , out_channels , kernel_size=2, stride=2)
    
    def forward(self, x):

        x = self.up(x)
        
        return x 
