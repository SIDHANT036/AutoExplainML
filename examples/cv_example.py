import torch
from autoexplainml import explain

# fake image tensor (1,3,224,224)
image = torch.randn(1, 3, 224, 224)

# simple CNN
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1)
)

result = explain(model, image, task="cv")

print(result)