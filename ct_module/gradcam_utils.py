# ============================================================
# NEURO.AI - GRAD-CAM UTILITY
# ============================================================

import torch
import numpy as np
import cv2


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
):
    """
    Generate a Grad-CAM heatmap for the predicted CT class.

    Parameters
    ----------
    model:
        StrokeNet model.

    image_tensor:
        Preprocessed image tensor with shape [1, 3, 224, 224].

    Returns
    -------
    cam:
        Normalized Grad-CAM heatmap of size 224 x 224.
    """

    model.eval()

    device = next(
        model.parameters()
    ).device

    gradients = []

    activations = []

    # ========================================================
    # TARGET LAYER
    # ========================================================

    target_layer = (
        model.model.layer4[-1]
    )

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output.detach()
        )

    # ========================================================
    # BACKWARD HOOK
    # ========================================================

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0].detach()
        )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        # ====================================================
        # PREPARE IMAGE
        # ====================================================

        image_tensor = image_tensor.to(
            device
        )

        image_tensor.requires_grad_(
            True
        )

        # ====================================================
        # FORWARD PASS
        # ====================================================

        output = model(
            image_tensor
        )

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

        # ====================================================
        # BACKWARD PASS
        # ====================================================

        model.zero_grad()

        output[
            0,
            predicted_class
        ].backward()

        # ====================================================
        # GET ACTIVATIONS + GRADIENTS
        # ====================================================

        grads = gradients[0][0]

        acts = activations[0][0]

        # ====================================================
        # GLOBAL AVERAGE POOLING
        # ====================================================

        weights = torch.mean(
            grads,
            dim=(1, 2)
        )

        # ====================================================
        # CREATE CAM
        # ====================================================

        cam = torch.zeros(
            acts.shape[1:],
            dtype=torch.float32,
            device=device
        )

        for i, weight in enumerate(
            weights
        ):

            cam += (
                weight * acts[i]
            )

        # ====================================================
        # RELU
        # ====================================================

        cam = torch.relu(
            cam
        )

        # ====================================================
        # NORMALIZE
        # ========================================================

        cam_min = cam.min()

        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        # ====================================================
        # CPU
        # ====================================================

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # RESIZE
        # ====================================================

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        # ====================================================
        # SMOOTH
        # ====================================================

        cam = cv2.GaussianBlur(
            cam,
            (7, 7),
            0
        )

        # ====================================================
        # THRESHOLD
        # ====================================================

        cam[
            cam < 0.25
        ] = 0

        return cam

    finally:

        # ====================================================
        # REMOVE HOOKS
        # ====================================================

        forward_handle.remove()

        backward_handle.remove()