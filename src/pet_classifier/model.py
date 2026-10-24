import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model():

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    

def main():
    pass

if __name__ == "__main__":
    main()