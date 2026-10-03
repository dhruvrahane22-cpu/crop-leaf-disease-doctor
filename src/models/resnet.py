import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes=38, mode='frozen'):
    """
    mode: 'frozen' | 'last_block' | 'full'
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    if mode == 'frozen':
        for param in model.parameters():
            param.requires_grad = False
    elif mode == 'last_block':
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode == 'full':
        for param in model.parameters():
            param.requires_grad = True
            
    # Replace final classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model