import re
from data.dataset import EtMirADatasetInference
import click
import os
import cv2
import torch
from tqdm import tqdm

import numpy as np
from user_config import DIR_ROOT
from config import CHANNELS, DIR_GT, DIR_MASKS, MEANS, STDS, SAMPLE_SIZE
from models.unet import UNet
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from data.data_instance import DatasetInstance
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar

from config import REGEX_MASK
from utils.utils import create_weight_map, get_iou, get_pf, get_avg_fun_over_set, get_predicted_mask, get_soft_image, visualize_overlap

@click.command()
@click.option('--device', default=0, help='accelarator to train on')
@click.option('--ckpt', help='path to checkpoint', required=True)
@click.option('--eval_mirrors', default=['ANSA-VI-1700_R'], help='mirrors to run inference on')
def eval_model(device, ckpt, eval_mirrors):

    if os.path.exists(ckpt) and ".ckpt" in ckpt:
        print("Resume training from checkpoint")
    else:
        raise FileNotFoundError(f"Could not locate checkpoint at {ckpt}")

    DIR_EVAL = os.path.join(*ckpt.split("/")[:-1], 'out')
    if not os.path.exists(DIR_EVAL):
        os.makedirs(DIR_EVAL)

    device = f'cuda:{device}'
    patch_size = None
    try:
        patch_size = torch.load(ckpt)['patch_size']
    except:
        patch_size = 512
    unet = UNet.load_from_checkpoint(
        ckpt, map_location=torch.device(device), strict=False)
    unet.to(device)
    unet.eval()

    weight_map = create_weight_map(0.5)

    samples = []

    for valid_mirror in sorted(os.listdir(DIR_MASKS)):
        res = re.search(REGEX_MASK, valid_mirror)
        if res:
            view_id = res.group(1)
            if view_id not in eval_mirrors:
                continue
            try:
                f_gt = os.path.join(DIR_GT, valid_mirror)
                assert os.path.exists(f_gt)
            except:
                f_gt = None
            f_mask = os.path.join(DIR_MASKS, f"{view_id}_mask.png")
            assert os.path.exists(f_mask)
            obj_id = res.group(2)
            dir_obj = os.path.join(DIR_ROOT, obj_id, 'PS')
            if os.path.exists(dir_obj):
                f_normals = os.path.join(dir_obj, f'{view_id}_N.tif')
                f_albedo = os.path.join(dir_obj, f'{view_id}_RHO.tif')
                f_depth = os.path.join(dir_obj, f'{view_id}_U.tif')
                if os.path.isfile(f_normals) and os.path.isfile(f_albedo) and os.path.isfile(f_depth):
                    instance = DatasetInstance(id=view_id,
                                               file_gt=None,
                                               file_normals=f_normals,
                                               file_albedo=f_albedo,
                                               file_depth=f_depth)
                    print(f'Create instance for {view_id}')
                else:
                    print(f"Could not locate normal, albedo, or depth map for mirror {view_id}")
                    continue
            else:
                raise FileNotFoundError(
                    f"Could not locate PS for mirror at {dir_obj}")

        dataset = DataLoader(EtMirADatasetInference(
            instance, transform_input=Compose([ToTensor(), Resize((SAMPLE_SIZE, SAMPLE_SIZE), antialias=True), Normalize(mean=[MEANS[enum] for enum in CHANNELS], std=[STDS[enum] for enum in CHANNELS])]), patch_size=patch_size), batch_size=256, shuffle=False, prefetch_factor=2, num_workers=8)
        if f_gt:
            gt = cv2.imread(f_gt, cv2.IMREAD_UNCHANGED)[
                :, :, 3].astype(bool)
        else:
            gt = None
        mask = cv2.imread(f_mask, cv2.IMREAD_UNCHANGED)[
            :, :, 2].astype(bool)

        image = get_predicted_mask(
            device, unet, weight_map, dataset, patch_size)

        orig_dim, image_soft = get_soft_image(dataset, image)
        image_soft[~mask] = 0
        cv2.imwrite(os.path.join(
            DIR_EVAL, f"{view_id}_soft.png"), (image_soft).astype(np.uint8))

        image_soft = image_soft/255.0
        if gt is not None:
            print(f'{view_id} IoU_T:\t{get_iou(image_soft, gt)}')
            print(f'{view_id} pF-Measure_T:\t{get_pf(image_soft, gt)}')

            samples = [(view_id, image_soft, gt)]
            res = minimize_scalar(get_avg_fun_over_set, bounds=(
                0, 1), args=(samples, get_iou), options={'disp': True})
            iou = -1*res['fun']
            print(f'{view_id} IoU^*:\t{iou}')

            res = minimize_scalar(get_avg_fun_over_set, bounds=(
                0, 1), args=(samples, get_pf), options={'disp': True})
            pf = -1*res['fun']
            print(f'{view_id} pF-Measure^*:\t{pf}')
            threshold = res.x.item()
        else:
            threshold = 0.5
        image_hard = np.where(image_soft >= threshold, np.ones(
            orig_dim, dtype=bool), np.zeros(orig_dim, dtype=bool))
        cv2.imwrite(os.path.join(
            DIR_EVAL, f"{view_id}_drawings.png"), (image_hard*255).astype(np.uint8))

        if gt is not None:
            image_error = visualize_overlap(
                image_hard, gt)
            cv2.imwrite(os.path.join(
                DIR_EVAL, f"{view_id}_error.png"), image_error)
        print('DONE')

if __name__ == '__main__':
    eval_model()
