import darmo
import torch
from collections import OrderedDict

model = darmo.create_model("relative_nas", num_classes=1000, pretrained=True)
model.reset_classifier(num_classes=100, dropout=0.2)
print(model)

# state_dict = torch.load("ImageNet.pth.tar", map_location='cpu')['state_dict']

# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]
#     new_state_dict[name] = v

# model.load_state_dict(new_state_dict, strict=True)
# torch.save(model.state_dict(), "pdarts.pt")