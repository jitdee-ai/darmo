import darmo

model = darmo.create_model("nasnet", num_classes=1000, pretrained=True)
model.reset_classifier(num_classes=100, dropout=0.2)
#print(model)