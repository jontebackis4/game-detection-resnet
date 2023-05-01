import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV files
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_to_idx, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.label_to_idx[self.data_frame.iloc[idx, 1]]
        return image, label

def modify_resnet18(num_classes, device):
    resnet18 = models.resnet18(weights='IMAGENET1K_V1')
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    resnet18 = resnet18.to(device)

    return resnet18

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

    return model, train_loss, val_loss, train_acc, val_acc

def plot_training_results(train_loss, val_loss, train_acc, val_acc):
    num_epochs = len(train_loss)
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def test_model(num_classes, device, dataloader):
    print('testing model...')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load("fine_tuned_resnet18.pth"))
    model.eval()

    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the fine-tuned model on the test dataset: {accuracy}%")


    


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img_dir = 'dataset/frames'
    train_csv = pd.read_csv('dataset/train_set.csv')
    val_csv = pd.read_csv('dataset/val_set.csv')
    unique_labels = sorted(pd.concat([train_csv.iloc[:, 1], val_csv.iloc[:, 1]]).unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    image_datasets = {
        'train': ImageDataset(csv_file='dataset/train_set.csv', root_dir=img_dir, label_to_idx=label_to_idx, transform=data_transforms['train']),
        'val': ImageDataset(csv_file='dataset/val_set.csv', root_dir=img_dir, label_to_idx=label_to_idx, transform=data_transforms['val']),
        'test': ImageDataset(csv_file='dataset/test_set.csv', root_dir=img_dir, label_to_idx=label_to_idx, transform=data_transforms['test']),
    }

    train_loader = DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    num_classes = 8
    model = modify_resnet18(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    model, train_loss, val_loss, train_acc, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=2, device=device)
    plot_training_results(train_loss, val_loss, train_acc, val_acc)
    torch.save(model.state_dict(), 'fine_tuned_resnet18.pth')

    # test_model(num_classes, device, dataloaders['test'])

if __name__ == '__main__':
    main()
