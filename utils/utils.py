import os
import cv2
import matplotlib.pyplot as plt
import scipy

import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.transforms import Compose, Normalize
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
from user_config import DIR_ROOT
from config import CHANNELS, MEANS, SAMPLE_SIZE, STDS

import tifffile

from data.items import EVAL_MIRRORS
from models.metrics import PseudoFMeasure


NORM_CONSTANT = 2**16-1
THRESHOLD = 0.3073

THRESHOLD_MIRRORS = EVAL_MIRRORS
UNNORMALIZE = Compose((Normalize(mean=[0 for _ in CHANNELS], std=[1/STDS[enum] for enum in CHANNELS]), Normalize(
    mean=[-MEANS[enum] for enum in CHANNELS], std=[1 for _ in CHANNELS])))
NORMALIZE = Normalize(mean=[MEANS[enum] for enum in CHANNELS], std=[
                      STDS[enum] for enum in CHANNELS])

metrics = torch.nn.ModuleDict({
    'jaccard_index': BinaryJaccardIndex(),
    'pf_measure': PseudoFMeasure(),
})


def read_tiff_file(file):
    image = (tifffile.imread(file)).astype(np.float32)
    return image


def create_weight_map(percent_of_boarder=0.5):
    """
    Create a weight map for a patch with weights decreasing from center to borders.

    Returns:
    numpy.ndarray: The weight map.
    """
    weight_map = np.ones((SAMPLE_SIZE, SAMPLE_SIZE))
    weight_map[0] = 0
    weight_map[-1] = 0
    weight_map[:, -1] = 0
    weight_map[:, 0] = 0
    weight_map = scipy.ndimage.distance_transform_edt(weight_map)
    weight_map = np.minimum(weight_map, percent_of_boarder*SAMPLE_SIZE//2)
    weight_map /= weight_map.max()
    weight_map = weight_map**2
   # plt.imshow(weight_map)
   # plt.savefig('weight_map.png')

    return weight_map


def get_green_transparent_image(image):
    # Create an empty RGBA image
    drawings = np.zeros(
        (image.shape[0], image.shape[1], 4), dtype=np.uint8)
    # Set the green color (R=0, G=255, B=0) for the foreground
    foreground_color = (0, 255, 0)
    # Set the Alpha channel to 255 for the object (fully opaque)
    drawings[image == 255, :] = (*foreground_color, 255)
    return drawings


def visualize_overlap(prediction_mask, ground_truth_mask):
    # Create RGB image with white background
    rgb_image = np.zeros(
        (prediction_mask.shape[0], prediction_mask.shape[1], 4), dtype=np.uint8)

    # Pixels that overlap in black
    overlap_indices = np.logical_and(
        prediction_mask, ground_truth_mask)
    rgb_image[overlap_indices] = [0, 0, 0, 255]  # Black

    # Pixels in prediction but not in ground truth (Green)
    false_positive_indices = np.logical_and(
        prediction_mask, ~ground_truth_mask)
    rgb_image[false_positive_indices] = [0, 255, 0, 255]  # Green

    # Pixels in ground truth but not in prediction (Red)
    false_negative_indices = np.logical_and(
        ~prediction_mask, ground_truth_mask)
    rgb_image[false_negative_indices] = [0, 0, 255, 255]  # Red

    return rgb_image


def get_avg_fun_over_set(threshold, samples, fun):
    threshold = threshold.item()
    runnin_mean = MeanMetric()
    for _, image_soft, gt in samples:
        runnin_mean.update(fun(image_soft, gt, threshold))
    return -runnin_mean.compute()


def get_iou(pred, target, threshold=0.5):
    if type(threshold) == torch.Tensor:
        threshold = threshold.item()
    return metrics['jaccard_index'](torch.Tensor(pred >= threshold).type(torch.bool), torch.Tensor(target).type(torch.bool))


def get_pf(pred, target, threshold=0.5):
    if type(threshold) == torch.Tensor:
        threshold = threshold.item()
    return metrics['pf_measure'](torch.Tensor(pred >= threshold).type(torch.bool), torch.Tensor(target).type(torch.bool))


def get_predicted_mask(device, unet, weight_map, dataset, patch_size=512):
    dim = np.array(
        dataset.dataset.inputs.shape[:-1])//(patch_size//SAMPLE_SIZE)
    preds = np.zeros(dim)
    counts = np.zeros(dim)
    with torch.no_grad():
        for batch in tqdm(dataset, total=len(dataset)):
            coords, inputs = batch
            coords = torch.stack(coords).numpy()
            outputs = unet(inputs.to(device))
            outputs = unet.aggregate_preds(outputs)
            outputs = outputs.cpu().numpy()
            for i in range(inputs.shape[0]):
                pred = outputs[i]
                coord = coords[:, i]//(patch_size//SAMPLE_SIZE)
                x_min, y_min = coord[1], coord[0]
                # only take last two dimensions (height x width)
                x_max, y_max = (pred.shape[-2:] + np.array([x_min, y_min]))
                temp_image = np.zeros(dim)
                temp_count = np.zeros(dim)
                # reduce weight of boarder pixels
                temp_image[x_min:x_max,
                           y_min:y_max] = pred * weight_map
                temp_count[x_min:x_max, y_min:y_max] = weight_map
                preds += temp_image
                counts += temp_count

        # Combine prediction
    np.seterr(all='ignore')
    image = torch.sigmoid(torch.Tensor(
        np.nan_to_num(preds/counts, nan=-np.inf))).numpy()
    np.seterr(all='warn')
    return image


def get_soft_image(dataset, image):
    orig_dim = np.array(dataset.dataset.inputs.shape[:-1])

    image_soft = (image*255).astype(np.uint8)
    image_soft = cv2.resize(
        image_soft, orig_dim[[1, 0]], interpolation=cv2.INTER_LINEAR)

    return orig_dim, image_soft


def get_paths(view_id, obj_id):
    dir_obj = os.path.join(DIR_ROOT, obj_id, 'PS')
    f_normals = os.path.join(dir_obj, f'{view_id}_N.tif')
    f_albedo = os.path.join(dir_obj, f'{view_id}_RHO.tif')
    f_depth = os.path.join(dir_obj, f'{view_id}_U.tif')
    return f_normals, f_albedo, f_depth
