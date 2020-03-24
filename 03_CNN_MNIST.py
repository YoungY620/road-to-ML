import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
# import matplotlib.pyplot as plt

torch.manual_seed(2)
# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = True   # set to False if you have downloaded
SHOW_GRAPH = False
'''
笔记:
    1.  train_data与test_data类型转化处理方式不同：
        train_data：   picture(PIL)--[toTensor]-->numpy.ndarray(range:(0,1))--[dataloader]-->Tensor
        test_data:     picture(PIL)--[type(FloatTensor)/255]-->Tensor(range:(0,1))
    2.  规定，每一笔train/test_data都要用list（[]）包起来
        如neural-net-regr.py回归问题中：
            x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # size(100)-->(100,1) 
        本例中：
            test_data = torchvision.datasets.MNIST(
                root='./mnist',
                train=False,
            )
            test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255 
            # size(2000,28,28)-->(2000,1,28,28)
        本例中(28, 28)视作一笔
    3.  learning rate过大会导致准确率一直在0.1徘徊!!!
        推荐lr=0.001
'''
# load train_data (ans download the mnist)
train_data = torchvision.datasets.MNIST(
    root="./datasets/mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),    # 一个函数转换。传入一个pil图像，返回转换后的版本
    download=DOWNLOAD_MNIST                         # 为True，且root指定路径下未下载，则进行下载
)
# print(train_data)
# loader for batch training 
train_loader = Data.DataLoader(
    dataset=train_data,             #
    batch_size=BATCH_SIZE,          # 
    shuffle=False                   # 是否随机打乱在进行下一次分割
)


# print("train_data:          ",type(train_data),"        size:",len(train_data))
# # train_data:           <class 'torchvision.datasets.mnist.MNIST'> 	size: 60000
# print("train_data.data:     ",type(train_data.data),"       size:",(train_data.data.size()))
# # train_data.data:      <class 'torch.Tensor'> 		size: torch.Size([60000, 28, 28])
# print("train_.targets:      ",type(train_data.targets),"        size:",(train_data.targets.size()))
# # train_.targets:       <class 'torch.Tensor'> 	size: torch.Size([60000])

# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
# print("train_loader:        ",type(train_loader))


# load test_data
test_data = torchvision.datasets.MNIST(
    root='./datasets/mnist',
    train=False,
)
test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255   
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 一个神经网络简介写法，一个卷积层
                                            # input shape: (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ), 
            nn.ReLU(),                      # 激活层
            nn.MaxPool2d(kernel_size=2)     # choose the max to represent the 2*2 area,
        )                                   # 输出shape：（16，14, 14）
        self.conv2 = nn.Sequential(         # input shape: (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )                                   # output shape: (32, 7, 7)
        self.all_connected = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.all_connected(x)
        return output, x


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# # following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):    # gives batch data, normalize x when iterate train_loader
        # training process
        output = cnn(b_x)[0]            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        # training finish

        if step % 100 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))

            print('\nEpoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)
            # if HAS_SK and SHOW_GRAPH:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
