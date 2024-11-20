from .contrastive.contrastive_loss import *
from .contrastive.contrastive_network import *

from .encoders.lstmnet import LSTMNet
from .encoders.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from .encoders.vit import ViTEncoder
from .encoders.wide_resnet import WideResNet

def load_encoder(encoder:str="ResNet", dataset='image'):
    if encoder == 'ResNet':
        if dataset == 'temp':
            return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], input_channels=3)
        else:
            # ResNet50
            return ResNet(block=Bottleneck, layers=[3, 4, 6, 3])        

    elif encoder == 'LSTMNet':
        return LSTMNet()
    
    elif encoder == 'ViT':
        return ViTEncoder()
    
    elif encoder == 'WideResNet':
        return WideResNet()