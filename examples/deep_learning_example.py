import torch
from autoexplainml import explain

# dummy tensor input
X = torch.randn(1, 10)

# dummy model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)

result = explain(model, X, task="deep_learning")

print(result)