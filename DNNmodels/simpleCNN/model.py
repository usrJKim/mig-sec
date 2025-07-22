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
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to("cuda")
elif args.model == "vgg19":
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to("cuda")
elif args.model == "alexnet":
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to("cuda")
elif args.model == "densenet":
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to("cuda")
elif args.model == "mobilenet":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to("cuda")

model.eval()
x = torch.randn(64, 3, 224, 224).to("cuda") #batch = 8
with torch.no_grad():
    output = model(x)
print(output.device)
print("DONE!")

