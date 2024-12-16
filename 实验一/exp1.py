import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import time

# 1. 数据集类
class MNISTDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.data = []
        if is_train:
            for label in range(10):
                label_dir = os.path.join(root_dir, "training", str(label))
                for img_name in os.listdir(label_dir):
                    self.data.append((os.path.join(label_dir, img_name), label))
        else:
            test_dir = os.path.join(root_dir, "our_testing_without_labels")
            for img_name in os.listdir(test_dir):
                self.data.append((os.path.join(test_dir, img_name), img_name))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label

# 2. 模型定义
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 3. 训练函数
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = MNISTNet().to(device)
    train_dataset = MNISTDataset("minist_dataset", is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 5
    
    print("开始训练...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{n_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{n_epochs} 完成 - '
              f'平均损失: {avg_loss:.6f}, '
              f'用时: {epoch_time:.2f}秒')
    
    print("训练完成！")
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("模型已保存为 mnist_model.pth")

# 4. 预测函数
def predict_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    test_dataset = MNISTDataset("minist_dataset", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = []
    print("开始预测...")
    
    with torch.no_grad():
        for data, img_names in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            
            for idx, p in enumerate(pred):
                results.append({
                    'image_name': img_names[idx],
                    'predicted_num': p.item()
                })
    
    df = pd.DataFrame(results)
    df.to_csv('minist_dataset/minist_result.csv', index=False)
    
    print(f"预测完成！共预测{len(results)}张图片")
    print("预测结果已保存到 minist_dataset/minist_result.csv")
    print("\n前5个预测结果:")
    print(df.head())

# 5. 评估函数
def evaluate_on_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    train_dataset = MNISTDataset("minist_dataset", is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    print("开始评估...")
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1].squeeze()
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\n模型在训练集上的准确率: {accuracy:.2f}%")
    
    for digit in range(10):
        mask = np.array(true_labels) == digit
        if mask.sum() > 0:
            digit_accuracy = 100. * np.sum(np.array(predictions)[mask] == digit) / mask.sum()
            print(f"数字 {digit} 的准确率: {digit_accuracy:.2f}%")
    
    return predictions, true_labels

if __name__ == "__main__":
    # 1. 训练模型
    train_model()
    
    # 2. 在测试集上预测并保存结果
    predict_and_save()
    
    # 3. 评估模型在训练集上的表现
    evaluate_on_training()