import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt 

'''
笔记：
    1.  numpy.ndarray或tensor之间使用==，返回一个元素为bool的ndarray或tensor
        如：
            a = np.array([8,9,3,6,5,7])
            b = np.array([8,9,1,1,5,7])
            print(a==b)                                     
                # [ True  True False False  True  True]
            print(torch.from_numpy(a)==torch.from_numpy(b)) 
                # tensor([ True,  True, False, False,  True,  True])
    3.  ndarray与tensor求和：
            x = torch.randn(2, 3, 1)
            print(x)
            print(x.sum(0)) # 变为3x1
            print(x.sum(1)) # 变为2x1
            print(x.sum(2)) # 变为2x3
            print(x.sum())  # 总和
            print(torch.sum(x))   # 总和
            print(torch.sum(x, 0))# 第0维求和 变为3x1
            print(torch.sum(x, 1))# 第1维求和 变为2x1
            print(torch.sum(x, 2))# 第2维求和 变为2x3
        ndarray同理，将torch换为np
    4.  accuracy的计算原理：将（pred_y==y）的返回值（tensor<bool>）转化为（tensor<int>），
        并求和，得正确结果数。除以总数即为正确率
'''


torch.manual_seed(2)

DOWNLOAD_MNIST = False
LR = 0.01
BATCH_SIZE = 50
EPOCH = 1
INPUT_SIZE = 28
TEST_SIZE = 2000

train_data = datasets.MNIST(
    root='./datasets/mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_data = datasets.MNIST(root='./datasets/mnist', train=False, )
test_x = test_data.data.type(torch.FloatTensor)[:TEST_SIZE]/255
test_y = test_data.targets[:TEST_SIZE]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,  
            hidden_size=64,         # rnn hidden unit, hidden_state特征数，大概与output_size相同
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn=RNN()
print(rnn)
optimizer = torch.optim.Adam(params=rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for index, (x, y) in enumerate(train_loader):
        # print("x: ",x.size())
        # print("y: ",y.size())         # output>> y:  torch.Size([50])
        x = x.view(-1, 28, 28)          # reshape, (batch_size,1,28,28)-->(batch_size,28,28)
        y_out = rnn(x)                  # one-hot output: torch.Size([50, 10])
        # print("y_out: ",y_out.size())   
        optimizer.zero_grad()
        loss = loss_func(y_out,y)       # auto convert one-hot to 0~9
        loss.backward()
        optimizer.step()

        if index%10==0:
            pred_y = torch.max(rnn(test_x),dim=1)[1]
            accuracy = float((pred_y==test_y).sum())/TEST_SIZE
            print("epoch: ",epoch,"|loss: %.4f"%loss, "|accuracy: %.4f"%accuracy)
