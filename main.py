"""Demo PyTorch MNIST model for the Seasalt.ai technical challenge."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 5 == 0:
            print(
                "Train epoch {} ({:.0f}%)\t Loss: {:.6f}".format(
                    epoch, 100.0 * idx / len(train_loader), loss.item(),
                ),
            )
            # print(
            #     'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, idx * len(data), len(train_loader.dataset),
            #         100. * idx / len(train_loader), loss.item(),
            #     ),
            # )
    print("train data size: ", len(train_loader))


def test(model, device, loader, optimizer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    print(
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(loader.dataset),
            100.0 * correct / len(loader.dataset),
        ),
    )
    print("test data size: ", len(loader.dataset))


# traning and test model parameters
# use batch_size_train and n_epoches to determine the training size
# for minimal training test set: batch_size_train = 64, n_eporch = 1
# total train data =
# (len(train_loader.dataset) / batch_size_train) * n_eporch = 938


n_epochs = 1
batch_size_train = 16
batch_size_test = 16
learning_rate = 0.01
log_interval = 5

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="./input",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./input",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)

model = Net().to(torch.device("cpu"))

optimizer = optim.SGD(model.parameters(), lr=0.01)

# execution time for me to learn with the mnist model
start_time = time.time()

for epoch in range(1, n_epochs + 1):
    train(model, torch.device("cpu"), train_loader, optimizer, epoch)

print("Train time: ", time.time() - start_time)

start_time = time.time()

test(model, torch.device("cpu"), test_loader, optimizer, epoch)

print("Test time: ", time.time() - start_time)

torch.save(model.state_dict(), "mnist_model.pth")
