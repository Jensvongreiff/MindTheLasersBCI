import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    Native PyTorch Implementation of EEGNet (Lawhern et al., 2018).
    Compact CNN for BCI robust to spatial non-stationarity.
    """
    def __init__(self, nb_classes=2, Chans=22, Samples=1000, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution & Spatial Depthwise Convolution
        self.block1 = nn.Sequential(
            # Temporal Filter
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            # Spatial Depthwise Filter (groups=F1 makes it depthwise)
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            # Depthwise portion of separable
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            # Pointwise portion of separable
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # Dummy pass to calculate Flatten size dynamically
        with torch.no_grad():
            dummy_out = self.block2(self.block1(torch.zeros(1, 1, Chans, Samples)))
            n_flat = dummy_out.view(1, -1).size(1)
            
        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, nb_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x