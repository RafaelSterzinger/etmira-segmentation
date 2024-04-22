import cv2
import numpy as np
from PIL import Image
import torch


def unnormalize_image(image_tensor):
    image_tensor = image_tensor.cpu()
    image_tensor -= image_tensor.min()
    max = image_tensor.max()
    image_tensor /= 1 if max == 0 else max
    return image_tensor


def visualize_overlap(prediction_mask, ground_truth_mask):
    # Create RGB image with white background
    rgb_image = 255*np.ones(
        (prediction_mask.shape[0], prediction_mask.shape[1], 3), dtype=np.uint8)

    # Pixels that overlap in black
    overlap_indices = np.logical_and(
        prediction_mask, ground_truth_mask)
    rgb_image[overlap_indices] = [0, 0, 0]  # Black

    # Pixels in prediction but not in ground truth (Green)
    false_positive_indices = np.logical_and(
        prediction_mask, ~ground_truth_mask)
    rgb_image[false_positive_indices] = [0, 255, 0]  # Green

    # Pixels in ground truth but not in prediction (Red)
    false_negative_indices = np.logical_and(
        ~prediction_mask, ground_truth_mask)
    rgb_image[false_negative_indices] = [255, 0, 0]  # Red

    return rgb_image.transpose(2, 0, 1)


def tensor_to_PIL(image_tensor):
    image_array = (image_tensor *
                   255).astype(np.uint8).squeeze()
    return Image.fromarray(image_array if len(image_array.shape) == 2 else np.transpose(image_array, (1, 2, 0)), mode='L' if len(image_array.shape) == 2 else 'RGB')


def log_image(x, target, preds):
    x = stack_batch(x)
    y_gt = stack_batch(target)
    y_hat = stack_batch(preds)
    bool_mask_pred = (preds >= 0.5).cpu().numpy().astype(bool)
    bool_mask_gt = (target >= 0.5).cpu().numpy().astype(bool)
    temp = torch.tensor(np.array([visualize_overlap(bool_mask_pred[i], bool_mask_gt[i])
                                  for i in range(bool_mask_gt.shape[0])]), dtype=float)
    error = stack_batch(temp)
    images = [x, y_gt, y_hat, error]
    stacked_image = Image.new(
        "RGB", (images[0].width, sum(image.height for image in images)))
    y_offset = 0
    for image in images:
        stacked_image.paste(image, (0, y_offset))
        y_offset += image.height
    return stacked_image


def stack_batch(image_tensor):
    images = [tensor_to_PIL(unnormalize_image(image_tensor[i]).numpy())
              for i in range(image_tensor.shape[0])]

    stacked_image = Image.new(
        'RGB', (sum(image.width for image in images), images[0].height))

    x_offset = 0
    for image in images:
        # Convert grayscale (L) images to RGB if needed
        if image.mode == 'L':
            image = image.convert('RGB')

        stacked_image.paste(image, (x_offset, 0))
        x_offset += image.width
    return stacked_image
