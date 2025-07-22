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
    model = models.resnet18(pretrained=False).cuda()
elif args.model == "vgg19":
    model = models.vgg19(pretrained=False).cuda()
elif args.model == "alexnet":
    model = models.alexnet(pretrained=False).cuda()
elif args.model == "densenet":
    model = models.densenet121(pretrained=False).cuda()
elif args.model == "mobilenet":
    model = models.mobilenet_v2(pretrained=False).cuda()

model.eval()
x = torch.randn(8, 3, 224, 224) #batch = 8
with torch.no_grad():
    output = model(x)
print("DONE!")

