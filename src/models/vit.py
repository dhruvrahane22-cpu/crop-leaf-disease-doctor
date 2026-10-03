import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit_b16(num_classes=38):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    
    # Replace final head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model