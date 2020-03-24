import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F 


# fake data
torch.manual_seed(2)
x_data = torch.ones(200,2)
x0 = torch.normal(2*x_data, 1)
x1 = torch.normal(-2*x_data, 1)
y0 = torch.ones(200)
y1 = torch.zeros(200)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1),).type(torch.LongTensor)

# show fake data
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(2, 10, 2)
print(net) # building success

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
for t in range(101):
    out = net(x) 
    print(x.size())                # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    if t % 10 == 0 or t in [3, 6]:
        # plot and show learning process
        plt.cla()
        _, prediction = torch.max(F.softmax(out), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=50, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/400.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color':  'red'})
        # plt.show()
        plt.pause(0.3)
plt.ioff()
plt.show()
