# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月11日
"""

from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(28 * 28 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x

