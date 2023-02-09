import torch
from base import BaseModel

class ConvUnit(BaseModel):
    
    def __init__(self, in_channels, out_channels, kernel_size, relu=True **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.norm = torch.nn.BatchNorm2d(out_channels)

        if relu:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = torch.nn.Sequential()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ResUnit(BaseModel):
    
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        if in_channels != out_channels:
            self.conv1 = ConvUnit(in_channels, out_channels, kernel_size, padding=1, stride=2)
            self.shortcut = ConvUnit(in_channels, out_channels, 1, stride=2)
        else:  
            self.conv1 = ConvUnit(in_channels, out_channels, kernel_size, padding=1)
            self.shortcut = torch.nn.Sequential()
            
        self.conv2 = ConvUnit(out_channels, out_channels, kernel_size, padding=1)
        self.relu = torch.nn.ReLU()
        
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        res = self.conv1(x)
        res = self.conv2(res)
        
        x = res + shortcut
        x = self.relu(x)
        
        return x

class ResNet34(BaseModel):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            ConvUnit(3, 64, 7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, 2, padding=1)
        )
        
        self.conv2_x = torch.nn.Sequential(
            ResUnit(64, 64, 3),
            ResUnit(64, 64, 3),
            ResUnit(64, 64, 3)
        )
        
        self.conv3_x = torch.nn.Sequential(
            ResUnit(64, 128, 3),
            ResUnit(128, 128, 3),
            ResUnit(128, 128, 3),
            ResUnit(128, 128, 3)
        )
        
        self.conv4_x = torch.nn.Sequential(
            ResUnit(128, 256, 3),
            ResUnit(256, 256, 3),
            ResUnit(256, 256, 3),
            ResUnit(256, 256, 3),
            ResUnit(256, 256, 3),
            ResUnit(256, 256, 3)
        )
        
        self.conv5_x = torch.nn.Sequential(
            ResUnit(256, 512, 3),
            ResUnit(512, 512, 3),
            ResUnit(512, 512, 3)
        )
        
        self.gap = torch.nn.Sequential(
              torch.nn.AvgPool2d(7,1)
        )
        
        self.fc = torch.nn.Sequential(
              torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x