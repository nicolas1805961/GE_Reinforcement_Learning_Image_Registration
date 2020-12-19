import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsummary


def accuracy(y_true, y_pred):
    acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y_true, dim=1))
    return acc / y_pred.size(0)


def choicem(y_true, y_pred):
    max_pred = torch.eq(y_pred, torch.max(y_pred, dim=1, keepdims=True).values)
    max_true = torch.eq(y_true, torch.max(y_true, dim=1, keepdims=True).values)

    diff = max_true.type(torch.int8) - max_pred.type(torch.int8)

    print(torch.ge(diff, 0).all(dim=1))
    print(torch.ge(diff, 0).all(dim=1).shape)

    # return torch.ge(diff, 0).all(dim=1) / y_pred.size(0)
    return 0


class DQN(nn.Module):

    def __init__(self, depth, height, width, outputs, device='cuda'):
        super(DQN, self).__init__()

        self.input_size = (depth, height, width)

        self.device = device

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 32, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(w, f=3, p=0, s=1):
            return ((w - f + 2*p) // s) + 1

        def maxpool_size_out(w):
            return w // 2

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(maxpool_size_out(conv2d_size_out(maxpool_size_out(conv2d_size_out(width)))))))
        linear_input_size = convw * convw * 128

        self.head1 = nn.Linear(linear_input_size, 512)
        self.head2 = nn.Linear(512, 512)
        self.head3 = nn.Linear(512, 64)
        self.head4 = nn.Linear(64, outputs)

        self.linear_input_size = linear_input_size

        self.to(device=device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.bn2(self.maxpool2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.head1(x))
        x = F.relu(self.head2(x))
        x = F.relu(self.head3(x))
        return self.head4(x)

    def fit(self, generator, optimizer=torch.optim.Adam, criterion=nn.MSELoss(), epochs=1):
        optimizer = optimizer(self.parameters())

        running_loss = 0.0

        for epoch in range(epochs):
            acc = 0.0
            choice = 0.0

            for index, batch in enumerate(generator):
                x = batch[0].to(self.device)
                q = batch[2].to(self.device)

                pred = self.__call__(x.float()).to(self.device)

                loss = criterion(pred, q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc += accuracy(q, pred)
                choice += choicem(q, pred)

            acc /= (index + 1)
            choice /= (index + 1)
            loss = running_loss / ((index + 1) * (epoch + 1))
            print(f'Epoch {epoch + 1}: loss {loss} accuracy {acc} choice {choice}')

    def summary(self):
        torchsummary.summary(self, self.input_size)
