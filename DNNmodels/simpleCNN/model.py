import torchvision.models as models
from torchvision import datasets, transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--mode", type=str, choices=["train","infer"],default="infer")
    parser.add_argument("--model", type=str, choices=["resnet", "vgg19", "alexnet", "densenet", "mobilenet"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="./data", help="Image dir path")

    return parser.parse_args()

def get_dataloader(image_dir, batch_size, mode="train"):
    transform = transforms.Compose([
                                       transforms.Resize((224,224)), # set input size
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], #ImageNet
                                                            std=[0.229, 0.224, 0.225])
                                   ])
    is_train = mode == 'train'
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
    return dataloader, len(dataset.classes)

def infer(model, dataloader):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to("cuda", non_blocking=True)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            print(f"Batch {idx+1}: Predicted labels = {pred.tolist()}")
            if idx >= 10: #infer only 10 batches
                break
    print("DONE!")

def train(model, dataloader, num_epochs):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to("cuda", non_blocking=True)
            targets= targets.to("cuda", non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}, batch_idx: {batch_idx}, Loss: {loss.item():.4f}")
    
if __name__ == "__main__":
    args = get_args()
    if args.mode == "infer":
        dataloader, num_classes = get_dataloader(args.data_dir, args.batch, mode="infer")
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
        infer(model, dataloader)
    
    elif args.mode == "train":
        dataloader, num_classes = get_dataloader(args.data_dir, args.batch, mode="train")
        if args.model == "resnet":
            model = models.resnet18(weights=None).to("cuda")
        elif args.model == "vgg19":
            model = models.vgg19(weights=None).to("cuda")
        elif args.model == "alexnet":
            model = models.alexnet(weights=None).to("cuda")
        elif args.model == "densenet":
            model = models.densenet121(weights=None).to("cuda")
        elif args.model == "mobilenet":
            model = models.mobilenet_v2(weights=None).to("cuda")
        train(model, dataloader, args.epochs)
