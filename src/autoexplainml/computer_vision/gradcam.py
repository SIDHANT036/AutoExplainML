# computer_vision/gradcam.py

import torch
import numpy as np

def explain_image(model, image_tensor):

    features = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal features
        features = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    last_conv = list(model.children())[-1]
    last_conv.register_forward_hook(forward_hook)
    last_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    output.backward()

    weights = gradients.mean(dim=[2,3], keepdim=True)
    cam = (weights * features).sum(dim=1)

    cam = cam.detach().numpy()

    return {
        "type": "gradcam",
        "heatmap": cam
    }