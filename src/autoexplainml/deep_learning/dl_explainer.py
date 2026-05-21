# deep_learning/dl_explainer.py

import torch

def explain_dl(model, input_tensor):

    input_tensor.requires_grad = True
    output = model(input_tensor)

    output.backward()

    gradients = input_tensor.grad.abs().mean(dim=0)

    return {
        "type": "deep_learning",
        "importance": gradients.detach().numpy()
    }