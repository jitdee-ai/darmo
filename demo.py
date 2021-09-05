import darmo

model = darmo.create_model("resnet50_1k", num_classes=1000, pretrained=True)
#model.reset_classifier(num_classes=100, dropout=0.2)
#print(model)


from torch.autograd import Variable
import torch
x_image = Variable(torch.randn(1, 3, 224, 224))
y = model(x_image)
print(y.size())