import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True)
a = 1
