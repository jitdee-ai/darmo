import darmo

model = darmo.create_model("pnas5", pretrained=True, auxiliary=False)
model.reset_classifier(num_classes=100, dropout=0.2)

model.eval()

from torch.autograd import Variable
import torch
x_image = Variable(torch.randn(1, 3, 224, 224))
y = model(x_image)
print(y.size())