import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from img_data.src.load_data import *

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 60
FORWARD_DAY = 5
BATCH_SIZE = 128
YEAR_START = 1993
YEAR_END = 1993
lr = 1e-5
IMAGE_DIR = '../monthly_20d'
LOG_DIR = 'logs'


class ConvNet_20day(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(8, 1)),
            # to make the output of convolution same size as input, padding size should be (67,1)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)))
        # self.fc1 = nn.Linear(?, 46080) or flatten?
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(46080, 2)
        self.drop = nn.Dropout1d(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        x = self.drop(x)
        return F.softmax(x, dim=1)

if __name__ == '__main__':
    model = ConvNet_20day()

    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    sample_data = StockImage(IMAGE_DIR,
                             YEAR_START,
                             YEAR_END,
                             transforms)

    sample_data = DataLoader(sample_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             drop_last=True)

    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    # sample training
    for epoch in range(20):
        running_loss = 0.0
        i = 0
        for data in sample_data:
            images, labels = data
            outputs = model(images)
            loss = loss_function(outputs, labels)
            # revise to zero gradient for every optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss
            i += 128
            if i % 1280 == 0:
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, 20, i/128, len(sample_data),
                                                          loss.item()))
        print(running_loss)


