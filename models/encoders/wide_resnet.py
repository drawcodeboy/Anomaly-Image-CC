from torchvision.models import wide_resnet50_2
from torch import nn

class WideResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = wide_resnet50_2()
        self.rep_dim = 128
        self.resnet.fc = nn.Linear(2048, self.rep_dim)
    
    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = WideResNet()
    print(model)