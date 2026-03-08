# """
# SquatNet — untrained PyTorch scoring model (scaffold only)
#
# This model is not used in the current pipeline. Scoring is handled
# by the rule-based weighted scorer in scoring.py.
#
# To activate: train on labeled squat data, save weights, load here,
# and call predict_score() from scoring.py instead of score_squat().
# Input contract: SquatFeatures.to_ml_vector() → 4-element float tensor.
#
# Commented out to avoid the ~2GB torch dependency in production.
# Uncomment and add torch to requirements.txt when ready to train. [torch>=2.3.0]
# """
#
# import torch
# import torch.nn as nn
#
#
# class SquatNet(nn.Module):
#
#     def __init__(self):
#
#         super().__init__()
#
#         self.model = nn.Sequential(
#
#             nn.Linear(4, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, x):
#
#         return self.model(x)
#
#
# model = SquatNet()
#
#
# def predict_score(features):
#
#     x = torch.tensor(features, dtype=torch.float32)
#
#     score = model(x).item()
#
#     return max(1, min(10, int(score)))