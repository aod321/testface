from icnnmodel import FaceModel, Stage2FaceModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def calc_centroid(tensor):
    # Inputs Shape(n, 9 , 64, 64)
    # Return Shape(n, 9 ,2)
    tensor = tensor.float() + 1e-10
    n, l, h, w = tensor.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = tensor.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    center_x = tensor.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    return torch.cat([center_y, center_x], 2)


class FirstStageModel(nn.Module):
    def __init__(self):
        super(FirstStageModel, self).__init__()
        self.first_model = FaceModel()
        # self.select_net = GetTheta()

        # Input' Shape must be (N, 9, 64, 64)
        self.locations_layer = nn.Sequential(
            nn.Conv2d(9, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # 8 x 32 x 32
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8 x 16 x 16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # 8 x 8 x 8
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8 x 4 x 4
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(128, 18),
                                nn.Sigmoid()
                                # nn.ReLU()
                                )

    # def get_theta(self):
    #     return self.select_net.get_theta()

    def forward(self, x):
        y = self.first_model(x)
        # y Shape(N, 9, 64, 64)
        # centroids = self.locations_layer(y)
        # centroids = centroids.view(-1, 128)
        # centroids = self.fc(centroids)
        # centroids = centroids.view(-1, 9, 2)
        # cen Shape(N, 9, 2)
        return F.sigmoid(y)


class SecondStageModel(nn.Module):
    def __init__(self):
        super(SecondStageModel, self).__init__()
        self.cen_to_theta = GetTheta()
        self.selectnet = SelectNet()
        self.stage2model = nn.ModuleList([Stage2FaceModel()
                                          for _ in range(6)])
        for i in range(5):
            self.stage2model[i].set_label_channels(1)
        self.stage2model[5].set_label_channels(3)

        self.theta = None

    def forward(self, x, orig):
        n = x.shape[0]
        self.theta = self.cen_to_theta(x)
        parts = self.selectnet(self.theta, orig)
        # Shape(N, 6, 3, 64, 64)
        out = []
        for i in range(6):
            out.append(self.stage2model[i](parts[:, i]))  # Shape(N, 1, 64, 64) or Shape(N, 3, 64, 64)
        out = torch.cat(out, dim=1)  # Shape (N, 8, 64, 64)
        assert out.shape == (n, 8, 64, 64)

        return out


class TwoStageModel(nn.Module):
    def __init__(self):
        super(TwoStageModel, self).__init__()
        self.first_stage = FirstStageModel()
        self.second_stage = SecondStageModel()

    def forward(self, x, orig):
        mask = self.first_stage(orig)
        cen = calc_centroid(mask) / 512.0
        out = self.second_stage(cen, orig)
        # Shape (N, 8, 64, 64)
        return out, cen


class GetTheta(nn.Module):
    def __init__(self):
        super(GetTheta, self).__init__()
        self.h = 512
        self.w = 512
        self.s = 64
        self.theta = None

    def get_theta(self):
        return self.theta

    def forward(self, cens):
        n, l, c = cens.shape
        # input Shape(N, 9, 2)
        out = cens * 512  # remap to Original
        # centroids Shape(N, 9, 2)
        mouth = torch.mean(out[:, 6:9], dim=1, keepdim=True)
        out = torch.cat([out[:, 1:6], mouth], dim=1)
        assert out.shape == (n, 6, 2)
        out[:, :, 0] = 256 - (8) * out[:, :, 0]
        out[:, :, 1] = 256 - (8) * out[:, :, 1]
        param = torch.zeros((n, 6, 2, 3)).to(cens.device)
        param[:, :, 0, 0] = 8
        param[:, :, 0, 2] = out[:, :, 0]
        param[:, :, 1, 1] = 8
        param[:, :, 1, 2] = out[:, :, 1]
        # Param Shape(N, 6, 2, 3)
        # Every label has a affine param
        ones = torch.tensor([[0., 0., 1.]]).repeat(n, 6, 1, 1).to(cens.device)
        param = torch.cat([param, ones], dim=2)
        param = torch.inverse(param)
        # ---               ---
        # Then, convert all the params to thetas
        self.theta = torch.zeros([n, 6, 2, 3]).to(cens.device)
        self.theta[:, :, 0, 0] = param[:, :, 0, 0]
        self.theta[:, :, 0, 1] = param[:, :, 0, 1] * 512 / 512
        self.theta[:, :, 0, 2] = param[:, :, 0, 2] * 2 / 512 + self.theta[:, :, 0, 0] + self.theta[:, :, 0, 1] - 1
        self.theta[:, :, 1, 0] = param[:, :, 1, 0] * 512 / 512
        self.theta[:, :, 1, 1] = param[:, :, 1, 1]
        self.theta[:, :, 1, 2] = param[:, :, 1, 2] * 2 / 512 + self.theta[:, :, 1, 0] + self.theta[:, :, 1, 1] - 1
        # theta Shape(N, 6, 2, 3)
        return self.theta


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()

    def forward(self, theta, orig):
        n, l, h, w = orig.shape
        theta_in = theta
        c = theta_in.shape[1]
        samples = []
        for i in range(c):
            grid = F.affine_grid(theta_in[:, i], [n, l, 64, 64], align_corners=True).to(theta.device)
            samples.append(F.grid_sample(input=orig, grid=grid, align_corners=True))
        samples = torch.stack(samples, dim=0)
        samples = samples.transpose(1, 0)
        assert samples.shape == (n, 6, 3, 64, 64)
        # Shape (N, c, 3, 64, 64)
        return samples
