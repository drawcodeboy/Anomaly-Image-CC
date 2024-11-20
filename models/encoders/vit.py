from torchvision.models import vit_b_16
from torch import nn

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16()
        self.rep_dim = 128
        self.vit.heads = nn.Linear(768, self.rep_dim)
    
    def forward(self, x):
        return self.vit(x)
    
if __name__ == '__main__':
    model = ViTEncoder()
    print(model)