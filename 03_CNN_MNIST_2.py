import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torch.utils.data as data

'''
#!!! learning rate过大会导致准确率一直在0.1徘徊!!!#
'''
BATCH_SIZE = 50
LR = 0.001          
EPOCH = 1
TEST_SIZE = 2000


# train_data mnist
train_data = MNIST(
    root='./datasets/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor()
)
# train_loader
train_loader = data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_data = MNIST(
    root='./datasets/mnist',
    train=False,
)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:TEST_SIZE]/255
test_y = test_data.targets.type(torch.LongTensor)[:TEST_SIZE]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.all_connect = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.all_connect(x)
        return x

cnn = CNN()
print("start training: \n",cnn)
print("#"*50)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for index, (x, y) in enumerate(train_loader):
        out = cnn(x)
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index%100 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))

            print('\nEpoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)