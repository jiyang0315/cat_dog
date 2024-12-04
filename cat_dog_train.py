import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms


# 设置随机数种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


# 以类的方式定义参数
class Args:
    def __init__(self)->None:
        self.batch_size = 4
        self.lr = 0.001
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')
args = Args()


# 神经网络
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool1
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool2
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool3
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # Fully connected layer 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # Fully connected layer 2
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),  # Fully connected layer 3 (output layer)
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        :param image_dir: 图片文件所在的目录
        :param label_file: 包含标签的文件，每行一个图像文件名和标签（例如，image1.jpg 0）
        :param transform: 可选的图像转换（比如数据增强）
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 加载标签文件，假设标签文件的格式是：每行包含图像文件名和标签，空格分隔
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        self.image_paths = []
        self.labels = []
        
        for line in lines:
            image_name, label = line.strip().split()
            self.image_paths.append(image_name)
            self.labels.append(int(label))  # 假设标签是整数

    def __len__(self):
        """返回数据集的大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """根据索引返回一个样本（图像和标签）"""
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        label = self.labels[idx]
        
        # 打开图像
        image = Image.open(image_path).convert('RGB')
        
        # 如果提供了转换（如数据增强、标准化等），应用转换
        if self.transform:
            image = self.transform(image)
        return image, label


def train():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据模型的输入要求调整大小
    transforms.ToTensor(),  # 转换为 Tensor，默认是 float32
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
    ])

    train_dataset = CustomDataset(image_dir="/home/jiyang/jiyang/Projects/Cat_Dog/cat_dog_dataset", label_file = '/home/jiyang/jiyang/Projects/Cat_Dog/train_labels.txt', transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CustomDataset(image_dir = "/home/jiyang/jiyang/Projects/Cat_Dog/cat_dog_dataset", label_file = '/home/jiyang/jiyang/Projects/Cat_Dog/test_labels.txt', transform=transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    model = AlexNet().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # =========================train=======================
        for idx, (images, label) in enumerate(tqdm(train_dataloader)):
            images = images.to(args.device).float()
            label = label.to(args.device)
            outputs = model(images)
            
            optimizer.zero_grad()
            loss = criterion(outputs, label)

            loss.backward()
# #             # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (images, label) in enumerate(tqdm(val_dataloader)):
                images = images.to(args.device).float()  # .to(torch.float)
                label = label.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))

if __name__=="__main__":
    train()
