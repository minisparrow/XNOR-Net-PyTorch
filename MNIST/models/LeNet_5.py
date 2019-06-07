import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F


class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, 5,padding=2)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16*5*5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
#     """
#     Input - 1x32x32
#     C1 - 6@28x28 (5x5 kernel)
#     tanh
#     S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
#     C3 - 16@10x10 (5x5 kernel, complicated shit)
#     tanh
#     S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
#     C5 - 120@1x1 (5x5 kernel)
#     F6 - 84
#     tanh
#     F7 - 10 (Output)
#     """
#     def __init__(self):
#         super(LeNet_5, self).__init__()
#         self.conv_1 = nn.Conv2d(1, 6, 5)
#         self.conv_2 = nn.Conv2d(6, 16, 5)
#         self.conv_3 = nn.Conv2d(16, 120,5)
#         self.fc_1 = nn.Linear(120, 84)
#         self.fc_2 = nn.Linear(84, 10)
# 
#         # self.convnet = nn.Sequential(OrderedDict([
#         #     ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
#         #     ('relu1', nn.ReLU()),
#         #     ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
#         #     ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
#         #     ('relu3', nn.ReLU()),
#         #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
#         #     ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
#         #     ('relu5', nn.ReLU())
#         # ]))
# 
#         # self.convnet = nn.Sequential(
#         #         nn.Conv2d(1,6,5),
#         #         nn.ReLU(),
#         #         nn.MaxPool2d(2,stride=2),
#         #         nn.Conv2d(6,16,5),
#         #         nn.ReLU(),
#         #         nn.MaxPool2d(2,stride=2),
#         #         nn.Conv2d(16,120,5),
#         #         nn.ReLU()
#         #         )
# 
#         # self.fc = nn.Sequential(OrderedDict([
#         #     ('f6', nn.Linear(120, 84)),
#         #     ('relu6', nn.ReLU()),
#         #     ('f7', nn.Linear(84, 10)),
#         #     ('sig7', nn.LogSoftmax(dim=-1))
#         # ]))
# 
#         # self.fc = nn.Sequential(
#         #         nn.Linear(120,84),
#         #         nn.ReLU(),
#         #         nn.Linear(84,10)
#         #         )
# 
#         # self.convnet = nn.Sequential(
#         #         nn.Conv2d(1,6,kernel_size=5,stride=1,padding=1),
#         #         nn.ReLU(),
#         #         nn.MaxPool2d(kernel_size=2,stride=2),
#         #         nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
#         #         nn.ReLU(),
#         #         nn.MaxPool2d(kernel_size=2,stride=2),
#         #         nn.Conv2d(16,120,kernel_size=5,stride=1,padding=0),
#         #         nn.ReLU()
#         #         )
#         # for m in self.modules():
#         #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         #         if hasattr(m.weight, 'data'):
#         #             m.weight.data.zero_().add_(1.0)
# 
#         return 
# 
#     def forward(self, img):
#         # for m in self.modules():
#         #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         #         if hasattr(m.weight, 'data'):
#         #             m.weight.data.clamp_(min=0.01)
# 
#         # output = self.convnet(img)
#         # output = output.view(img.size(0), -1)
#         # output = self.fc(output)
# 
#         out = F.relu(self.conv_1(img))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv_2(out))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv_3(out))
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc_1(out))
#         out = F.relu(self.fc_2(out))
#         return out 
