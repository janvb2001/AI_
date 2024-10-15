import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Image detector
class convolution1(nn.Module):
    def __init__(self):
        super(convolution1, self).__init__()

        self.conv_block = nn.Sequential(
            # input the desired layers here
            nn.LazyConv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.conv_block(x)


class DetectionHead1(nn.Module):
    def __init__(self, num_classes: int):
        super(DetectionHead1, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_cls = nn.Linear(512, num_classes)
        self.fc_bbox = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        cls_logits = self.fc_cls(x)
        bbox_pred = self.fc_bbox(x)
        return cls_logits, bbox_pred


class ImageDetector1(nn.Module):
    def __init__(self, num_classes: int):
        super(ImageDetector1, self).__init__()
        self.backbone = convolution1()
        self.head = DetectionHead1(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        cls_logits, bbox_pred = self.head(features)
        return cls_logits, bbox_pred

    def loss_fn(self, cls_logits, bbox_pred, cls_targets, bbox_targets):
        cls_loss = nn.functional.cross_entropy(cls_logits, cls_targets)
        bbox_loss = nn.functional.mse_loss(bbox_pred, bbox_targets)
        return cls_loss + bbox_loss


# Simplest
class simplestBackbone(nn.Module):
    def __init__(self):
        # super(simplestBackbone, self).__init__()
        super().__init__()
        self.conv1 = nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 128)
        # output = 9 * [x1, y1, conf1, x2, y2, conf2, x3, y3, conf3, x4, y4, conf4]
        self.fc3 = nn.Linear(128, 108)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def dist(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def loss_fn(self, x, gt):

        loss = nn.functional.mse_loss(x, gt, reduction="sum")

        # res = []
        # for i in range(9):
        #     if max(x[i * 12 + 2], x[i * 12 + 5], x[i * 12 + 8], x[i * 12 + 11]) >= 0.5:
        #         res.append(x[i * 12 : i * 12 + 12])

        # middles_res = []
        # for i in range(len(res)):
        #     xm = (res[i][0] + res[i][3] + res[i][6] + res[i][9]) / 4
        #     ym = (res[i][1] + res[i][4] + res[i][7] + res[i][10]) / 4
        #     middles_res.append([xm, ym])

        # # Dependent still on the format of gt
        # middles_gt = []
        # for i in range(len(gt)):
        #     xm = (gt[i][0] + gt[i][3] + gt[i][6] + gt[i][9]) / 4
        #     ym = (gt[i][1] + gt[i][4] + gt[i][7] + gt[i][10]) / 4
        #     middles_gt.append([xm, ym])

        # correspondence = np.empty(len(middles_gt))
        # for i in range(len(middles_gt)):
        #     minlen = -1
        #     minleni = -1
        #     for j in range(len(middles_res)):
        #         distfromcenter = self.dist(
        #             middles_gt[i][0],
        #             middles_gt[i][1],
        #             middles_res[j][0],
        #             middles_res[j][1],
        #         )
        #         if (
        #             distfromcenter < minlen or minlen == -1
        #         ) and j not in correspondence:
        #             minlen = distfromcenter
        #             minleni = j
        #     correspondence[i] = minleni

        # loss = 0

        # for i in range(len(correspondence)):
        #     loss += self.dist(
        #         gt[i][0], gt[i][1], res[correspondence[i]][0], res[correspondence[i]][1]
        #     )
        #     loss += self.dist(
        #         gt[i][3], gt[i][4], res[correspondence[i]][3], res[correspondence[i]][4]
        #     )
        #     loss += self.dist(
        #         gt[i][6], gt[i][7], res[correspondence[i]][6], res[correspondence[i]][7]
        #     )
        #     loss += self.dist(
        #         gt[i][9],
        #         gt[i][10],
        #         res[correspondence[i]][9],
        #         res[correspondence[i]][10],
        #     )

        #     loss += (res[correspondence[i]][2] - gt[i][2]) ** 2
        #     loss += (res[correspondence[i]][2] - gt[i][5]) ** 2
        #     loss += (res[correspondence[i]][2] - gt[i][8]) ** 2
        #     loss += (res[correspondence[i]][2] - gt[i][11]) ** 2

        # for i in range(len(res)):
        #     if i not in correspondence:
        #         loss += res[i][2] ** 2
        #         loss += res[i][5] ** 2
        #         loss += res[i][8] ** 2
        #         loss += res[i][11] ** 2

        return loss
