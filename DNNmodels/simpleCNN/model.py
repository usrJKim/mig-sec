import torchvision.models as models
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--model", type=str, choices=["resnet", "vgg19", "alexnet", "densenet", "mobilenet"])

    return parser.parse_args()


args = get_args()
if args.model == "resnet":
    model = models.resnet18(pretrained=False)
    #edit output layer
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
elif args.model == "vgg19":
    model = models.vgg19(pretrained=False)
    #edit output layer
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)
elif args.model == "alexnet":
    model = models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)
elif args.model == "densenet":
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, out_features=10)
elif args.model == "mobilenet":
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10)

# TODO pretrained=True

x = torch.randn(8, 3, 224, 224) #batch = 8
output = model(x)
print("DONE!")

