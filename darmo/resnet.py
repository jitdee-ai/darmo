from .registry import register_model
import torchvision.models as models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

url_cfgs = {
    'resnet50_21k' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/resnet50_miil_21k.pth',
}

@register_model
def resnet50_1k(pretrained=True, num_classes=1000, auxiliary=False):
    return ResNet(pretrained=pretrained, num_classes=num_classes, name="resnet50_1k")

@register_model
def resnet50_21k(pretrained=True, num_classes=11221, auxiliary=False):
    return ResNet(pretrained=pretrained, num_classes=11221, name="resnet50_21k")

class ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000, name="resnet50_21k"):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        self.num_features = 512 * 4

        if name == 'resnet50_21k':
            del self.resnet.fc.weight
            del self.resnet.fc.bias
            init = model_zoo.load_url(url_cfgs[name], progress=True, map_location='cpu')['state_dict']
            del init['fc.weight']
            del init['fc.bias']
            self.resnet.load_state_dict(init, strict=True)
            self.resnet.fc = nn.Linear(self.num_features, num_classes)
        print("loaded state dict")

    def forward(self, x):
        return self.resnet(x)

    def reset_classifier(self, num_classes, dropout=0.0):
        self.num_classes = num_classes
        #self.drop_rate = dropout

        del self.resnet.fc

        if self.num_classes:
            self.resnet.fc = nn.Linear(self.num_features, self.num_classes)
        else:
            self.resnet.fc = None