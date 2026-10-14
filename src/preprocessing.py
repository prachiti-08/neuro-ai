"""
Facial Palsy Image Preprocessing
================================

Inference preprocessing must match the evaluation preprocessing
used by the trained ResNet18 model.
"""

from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224

# ImageNet normalization used by the ResNet18 training pipeline.
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_inference_transform():
    """
    Preprocessing for validation/test/inference images.

    No random augmentation is used during inference.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def preprocess_image(image):
    """
    Convert a PIL image into a model-ready tensor.

    Returns:
        Tensor with shape [1, 3, 224, 224]
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL.Image.Image object."
        )

    image = image.convert("RGB")

    transform = get_inference_transform()

    tensor = transform(image)

    # Add batch dimension.
    tensor = tensor.unsqueeze(0)

    return tensor