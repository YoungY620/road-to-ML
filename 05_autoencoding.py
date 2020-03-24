import torch
import torchvision
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt 
import numpy as np 


torch.manual_seed(2)

LR = 0.01
BATCH_SIZE = 50
EPOCH = 1
TEST_N_IMG = 4
DOWNLOAD_MNIST = True
activ_func = nn.Tanh


train_data = datasets.MNIST(
    root='./datasets/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            activ_func(),
            nn.Linear(128, 32),
            activ_func(),
            nn.Linear(32, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            activ_func(),
            nn.Linear(32, 128),
            activ_func(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


autoencoder = Autoencoder()
print(autoencoder)

optimizer = torch.optim.Adam(params=autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss()

fig, ax = plt.subplots(2, TEST_N_IMG,figsize=(5, 2))
plt.ion()
test_img = datasets.MNIST(root='./datasets/mnist',train=False).data.type(torch.FloatTensor)
indexes = (torch.rand(size=(TEST_N_IMG,))*(test_img.size()[0])).type(torch.LongTensor)
test_img = test_img[indexes].view(-1,28*28)/225
for i in range(TEST_N_IMG):
    ax[0][i].imshow(np.reshape(test_img[i].numpy(),newshape=(28,28)), cmap='gray')
    ax[0][i].set_xticks(())
    ax[0][i].set_yticks(())

for epoch in range(EPOCH):
    for index, (x, y) in enumerate(train_loader):
        x = x.view(-1, 28*28)
        y = x.view(-1, 28*28)

        encode, decode = autoencoder(x)
        loss = loss_func(decode, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print("epoch:  ",epoch,"|index:  ",index,"|loss:  %6.4f"%loss)
            _,decode_img = autoencoder(test_img)
            print(decode_img.size())
            for i in range(TEST_N_IMG):
                ax[1][i].imshow(np.reshape(decode_img.detach().numpy()[i], newshape=(28,28)), cmap='gray')
                ax[1][i].set_xticks(())
                ax[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)
plt.ioff()
plt.show()