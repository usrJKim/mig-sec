import torchvision.models as models
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--mode", type=str, choices=["train","infer"],default="infer")
    parser.add_argument("--model", type=str, choices=["resnet", "vgg19", "alexnet", "densenet", "mobilenet"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iter", type=int, default=100)

    return parser.parse_args()

class FakeImageNetDataSet(Dataset):
    def __init__(self, length=50000, image_size=(3,244,244), num_classes=1000, transform=None):
        self.length=length
        self.image_size=image_size
        self.num_classes=num_classes
        self.transform=transform

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        #Gen from CPU
        img = torch.randn(self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()

        if self.transform:
            img = self.transform(img)
        return img, label

def infer(model, batch_size):
    
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224).to("cuda") #batch = 8
    with torch.no_grad():
        output = model(x)
    print(output.device)
    print("DONE!")

def train(model, num_epochs, num_iter, batch_size):
    #Make fake data
    fake_dataset = FakeImageNetDataset(length=num_epochs*num_iter)
    train_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to("cuda", non_blocking=True)
            targets= targets.to("cuda", non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1}, batch_idx: {batch_idx}")
    
if __name__ == "__main__":
    args = get_args()
    if args.mode == "infer":
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
        infer(model, args.batch)
    
    elif args.mode == "train":
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
        train(model, args.epochs, args.iter, args.batch)
