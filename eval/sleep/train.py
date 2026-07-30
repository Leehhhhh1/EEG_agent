import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.init as init
import numpy as np
import os

# -----------------------------
# 加载数据。
# -----------------------------
save_folder = "./data"
X_train = torch.load(f"{save_folder}/X_train.pt")
y_train = torch.load(f"{save_folder}/y_train.pt")
X_test = torch.load(f"{save_folder}/X_test.pt")
y_test = torch.load(f"{save_folder}/y_test.pt")



# -----------------------------
# 加载数据。
# -----------------------------
def get_dataloader(X, y, batch_size=64, shuffle=True):
    """获取 get dataloader 相关信息。"""
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

train_loader = get_dataloader(X_train, y_train)
test_loader = get_dataloader(X_test, y_test, batch_size=1024, shuffle=False)

# -----------------------------
# 3. 模型
# -----------------------------
import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution:
      - depthwise: groups = in_channels (spatial filtering per feature map)
      - pointwise: 1x1 conv to mix features
    """
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=False):
        """初始化对象状态。"""
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=(1,1),
                                   padding=0, bias=bias)
    def forward(self, x):
        """执行模型前向计算。"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet (Lawhern et al. 2018) style.
    Input shape: (batch, 1, n_channels, n_samples)
    """
    def __init__(self,
                 n_channels,
                 n_samples,
                 n_classes,
                 F1=32,         # 保留的开发备注。
                 D=8,          # 保留的开发备注。
                 F2=None,      # 保留的开发备注。
                 kernel_length=75,
                 dropout=0.,
                 pool_kernel=5,
                 verbose=False):
        """初始化对象状态。"""
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.verbose = verbose
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), bias=True)
        # 张量形状说明。
        self.bn1 = nn.InstanceNorm2d(F1, affine=True)

        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        # 张量形状说明。
        self.bn2 = nn.InstanceNorm2d(F1 * D, affine=True)
        self.activation = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_kernel))
        self.dropout1 = nn.Dropout(p=dropout)

        sep_kernel_length = kernel_length // pool_kernel
        self.sep = SeparableConv2d(F1 * D, F2, kernel_size=(1, sep_kernel_length), bias=False)
        # 张量形状说明。
        self.bn3 = nn.InstanceNorm2d(F2, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pool_kernel))
        self.dropout2 = nn.Dropout(p=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(F2, n_classes)

        self._initialize_weights()

    def _forward_features(self, x):
        # 保留的开发备注。
        """处理 forward features 相关逻辑。"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.pool(x)
        return x

    def forward(self, x):
        """执行模型前向计算。"""
        x = x.unsqueeze(1)
        x = self._forward_features(x).squeeze()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """处理 initialize weights 相关逻辑。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



# -----------------------------
# 4. 训练函数
# -----------------------------
import time
def train_model(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    """处理 train model 相关逻辑。"""
    model.to(device)
    # 保留的开发备注。
    # 保留的开发备注。
    # weights = weights / weights.sum() * len(class_counts)  # 归一化
    # 保留的开发备注。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds = []
        all_labels = []
        time0 = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
            
            # 收集预测和标签
            all_preds.append(outputs.argmax(1).detach().cpu())
            all_labels.append(y_batch.cpu())

        # 拼接所有 batch
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        time1 = time.time()
        print(f"Epoch {epoch+1}/{epochs}:{time1-time0:.2f}s - Loss: {total_loss/total:.4f} - Acc: {correct/total:.4f} - MF1: {macro_f1:.4f}")
        test_model(model, test_loader, device=device)

# -----------------------------
# 5. 测试函数
# -----------------------------
best_f1 = 0

def test_model(model, test_loader, device='cpu', class_names=None):
    """处理 test model 相关逻辑。"""
    global best_f1  
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Test Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    if macro_f1 > 0.74 and macro_f1 > best_f1:
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "./checkpoints/best_model.pth")
        print(f"Model saved! Acc={acc:.4f}, Macro-F1={macro_f1:.4f}")
        best_f1 = macro_f1

    return acc, macro_f1

# -----------------------------
# 6. 训练与测试
# -----------------------------
'''
LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}
'''

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
model = EEGNet(n_channels=2, n_samples=3000, n_classes=5)
train_model(model, train_loader, epochs=100, lr=5e-3, device=device)

# 86 82