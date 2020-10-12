import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import ContinuousConv
from util import neighbor_search


class Deepfluid(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc1 = nn.Linear(in_features=2, out_features=32)
        self.cconv1 = ContinuousConv(kernel_size=4, in_channel=2, out_channel=32)
        self.fc2 = nn.Linear(in_features=96, out_features=64)
        self.cconv2 = ContinuousConv(kernel_size=4, in_channel=96, out_channel=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.cconv3 = ContinuousConv(kernel_size=4, in_channel=64, out_channel=64)
        self.fc4 = nn.Linear(in_features=64, out_features=3)
        self.cconv4 = ContinuousConv(kernel_size=4, in_channel=64, out_channel=3)

    def forward(self, dy_positions, dy_feats, box_positions, box_feats):
        # First Layer: input --(CConv1, fc1)---> 32
        # continuous convolution with boxes
        box_indxs = neighbor_search(dy_positions, box_positions)

        # continuous convolution with dynamic particles
        dy_indxs = neighbor_search(dy_positions, dy_positions)

        dy_feats_1 = []    # map input_feats into feats in a new shape
        length = len(dy_positions)
        for i in range(length):
            # map to 32-d features
            box_cc = self.cconv1(dy_positions[i], box_positions, box_feats, box_indxs)    # 1*32
            dy_cc = self.cconv1(dy_positions[i], dy_positions, dy_feats, dy_indxs)   # 1*32
            self_feats = self.input_fc1(dy_feats[i])

            # concatenate to 96-dimension feature
            x = F.relu(torch.cat((box_cc, dy_cc, self_feats)))
            dy_feats_1.append(x)

        # Second Layer: 96 --(CConv2, fc2)--> 64
        dy_feats_2 = []
        for i in range(length):
            cc = self.cconv2(dy_positions[i], dy_positions, dy_feats_1, dy_indxs)   # 1*64
            cc = F.relu(cc)
            self_feats = self.fc2(dy_feats_1[i])
            dy_feats_2.append(cc + self_feats)

        # Third Layer: 64 --(CConv3, fc3, shortcut)--> 64







            cc = self.fc2(x)

            x = dy_feats[i] + cc    # a feature map of 32-dimension

            # map into 64-dimension feats with fc2
            self_feats = self.fc2(x)
            neighbor_feats = [self.fc2(each) for each in neighbor_feats]
            cc = self.cconv2(dy_positions[i], neighbor_positions, neighbor_feats)
            x = self_feats + cc

            # another 64 -> 64 mapping
            self_feats = self.fc3(x)
            neighbor_feats = [self.fc3(each) for each in neighbor_feats]
            cc = self.cconv3(dy_positions[i], neighbor_positions, neighbor_feats)
            x = self_feats + cc

            # output into 3-dimension delta position
            self_feats = self.fc4(x)
            neighbor_feats = [self.fc4(each) for each in neighbor_feats]
            cc = self.cconv4(dy_positions[i], neighbor_positions, neighbor_feats)
            x = self_feats + cc

            res. append(x)
        return res

