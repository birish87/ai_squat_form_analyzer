"""
Simple PyTorch model for squat quality scoring.

This demonstrates ML pipeline capability.
"""

import torch
import torch.nn as nn


class SquatNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):

        return self.model(x)


model = SquatNet()


def predict_score(features):

    x = torch.tensor(features, dtype=torch.float32)

    score = model(x).item()

    return max(1, min(10, int(score)))