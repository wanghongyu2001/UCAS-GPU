# 这是程序一的Python模板程序，您可以直接提交该程序而不进行任何修改（如果不介意准确率的分数），对于程序一的CUDA C/C++模板程序（加分题），请参考程序二的模板程序自行实现

'''
Package                  Version
------------------------ ----------
certifi                  2023.7.22
charset-normalizer       3.2.0
cmake                    3.27.4.1
filelock                 3.12.4
idna                     3.4
Jinja2                   3.1.2
lit                      16.0.6
MarkupSafe               2.1.3
mpmath                   1.3.0
networkx                 3.1
numpy                    1.26.0
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
Pillow                   10.0.1
pip                      23.2.1
requests                 2.31.0
setuptools               68.0.0
sympy                    1.12
torch                    2.0.1
torchaudio               2.0.2
torchvision              0.15.2
triton                   2.0.0
typing_extensions        4.7.1
urllib3                  2.0.4
wheel                    0.38.4
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os


# 定义LeNet模型 tanh
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, stride=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(3 * 6 * 6, 10)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        # x = self.pool(torch.tanh(self.conv2(x)))
        # x = x.view(-1, 16 * 4 * 4)
        x = x.view(-1, 3 * 6 * 6)
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        x = (self.fc1(x))
        # x = (self.fc2(x))
        # x = self.fc3(x)
        return x
# 定义LeNet模型
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建模型
model = LeNet()
model = model.to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

# 训练模型
for epoch in range(70):
    print('epoch ', epoch)
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()    
    print(correct/total) 

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(correct/total)  
fc1_weight = model.fc1.weight
fc1_bias = model.fc1.bias
# fc2_weight = model.fc2.weight
# fc2_bias = model.fc2.bias
# fc3_weight = model.fc3.weight
# fc3_bias = model.fc3.bias



# # 导出模型参数，也可以自定义导出模型参数的文件格式，这里使用了最简单的方法，但请注意，如果改动了必须保证程序二能够正常读取
for name, param in model.named_parameters():
    np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())
# 进行所需的运算
# result1 = fc3_weight 
# # result2 = fc3_weight @ fc2_weight @ fc1_bias + fc3_weight @ fc2_bias + fc3_bias

# # # 将结果保存到文本文件
# np.savetxt(os.path.join(script_dir, f'./fc1.weight.txt'), result1.detach().cpu().numpy().flatten())
# np.savetxt(os.path.join(script_dir, f'./fc1.bias.txt'), result2.detach().cpu().numpy().flatten())
# np.savetxt('fc4.weight.txt', result1.detach().cpu().numpy().flatten())
# np.savetxt('fc4.bias.txt', result2.detach().cpu().numpy().flatten())
