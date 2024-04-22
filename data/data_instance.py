from data.items import EVAL_MIRRORS
import albumentations as A
import os
from PIL import Image
import cv2
import numpy as np

from skimage.filters import gaussian
from tqdm import tqdm
from config import PATCH_SIZE_X, PATCH_SIZE_Y, STRIDE_X, STRIDE_Y, PREPROCESSING_PARAMS, TRAIN_PATH, UNLABELLED_TRAIN_PATH, VAL_PATH, MapType

from utils.utils import read_tiff_file

class DatasetInstance:
    def __init__(self,
                 id,
                 file_gt,
                 file_normals,
                 file_albedo,
                 file_depth,
                 has_label=True):

        self.id = id
        self.use_real = has_label
        self.file_gt = file_gt
        self.file_normals = file_normals
        self.file_albedo = file_albedo
        self.file_depth = file_depth
        self.data = {enum: None for enum in MapType}

    # for train data split into large chunks, for validation, leave as is
    def set_up_data(self):
        print(f"Create data for {self.id}")
        # load data
        self.data = {enum: self.get_map(enum) for enum in MapType}
        if self.is_in_validation():
            print(f"Create data for validation")
            inputs = self.get_inputs()
            gt = self.get_map(MapType.GT)

            def filename(name):
                filename = f"{self.id}_{name}.npy"
                path = os.path.join(
                    VAL_PATH, filename)
                return path
            np.save(filename('inputs'), inputs)
            np.save(filename('gt'), gt)
        else:
            print(f"Create data for training")
            # apply padding of 2 to get cleaner patches
            padding = A.PadIfNeeded(min_width=8964,
                                    min_height=6720)
            self.data = {enum: padding(image=self.get_map(enum))[
                'image'] for enum in MapType}
            coords = get_coords(patch_size_x=PATCH_SIZE_X, patch_size_y=PATCH_SIZE_Y,
                                stride_x=STRIDE_X, stride_y=STRIDE_Y, x_max=self.data[MapType.GT].shape[1], y_max=self.data[MapType.GT].shape[0])
            inputs = self.get_inputs()
            for _coords in tqdm(coords):
                def filename(name):
                    filename = f"{self.id}_{_coords[0]}_{_coords[1]}_{_coords[2]}_{_coords[3]}_{name}.npy"
                    path = os.path.join(
                        TRAIN_PATH if self.use_real else UNLABELLED_TRAIN_PATH, filename)
                    return path
                patch_inputs, gt = self.get_patch(_coords, inputs)
                if gt.sum() == 0:
                    continue
                np.save(filename('inputs'), patch_inputs)
                np.save(filename('gt'), gt)
        print("Done")

    def get_map(self, map_type: MapType):
        if map_type in (MapType.NX, MapType.NY, MapType.NZ):
            if self.data[map_type] is None:
                n = read_tiff_file(self.file_normals)
                # rescale vectors to unit
                norms = np.linalg.norm(n, axis=-1, keepdims=True)
                n /= norms
                self.data[MapType.NX] = n[:, :, 0]
                self.data[MapType.NY] = n[:, :, 1]
                self.data[MapType.NZ] = n[:, :, 2]
            return self.data[map_type]
        elif map_type == MapType.DEPTH:
            if self.data[map_type] is None:
                with Image.open(self.file_normals) as img:
                    ppmm = img.info.get('dpi')[0] / 25.
                d = read_tiff_file(self.file_depth)
                if 'depth_highpass_sigma_mm' in PREPROCESSING_PARAMS:
                    sigma = PREPROCESSING_PARAMS['depth_highpass_sigma_mm'] * ppmm
                    d = d - gaussian(d, sigma) + d.mean()
                if 'depth_normalize_stds' in PREPROCESSING_PARAMS:
                    d = (d - d.mean()) / d.std()
                    # cap outliers, i.e. >/< 3 sigma
                    d[d > 3] = 3
                    d[d < -3] = -3
                    d -= d.min()
                    d /= d.max()
                    d = d.astype(np.float32)
                self.data[map_type] = d
            return self.data[map_type]
        elif map_type == MapType.ALBEDO:
            if self.data[map_type] is None:
                a = read_tiff_file(self.file_albedo)
                a -= a.min()
                a /= a.max()
                self.data[map_type] = a
            return self.data[map_type]
        elif map_type == MapType.GT:
            if self.data[map_type] is None:
                gt = cv2.imread(self.file_gt, cv2.IMREAD_UNCHANGED)
                try:
                    gt = gt[:, :, 3]
                except:
                    gt = gt[:, :, 0]
                self.data[map_type] = gt.astype(bool).astype(np.float32)
            return self.data[map_type]
        else:
            raise ValueError(f'{map_type} is not a valid MapType.')

    def is_in_validation(self):
        return self.id in EVAL_MIRRORS

    def get_inputs(self):
        return np.dstack([self.get_map(c) for c in MapType.get_inputs_enums()])

    def get_patch(self, coords, inputs):
        x1, y1, x2, y2 = coords
        patch_gt = self.get_patch_gt(x1, y1, x2, y2)
        patch_inputs = self.get_patch_input(inputs, x1, y1, x2, y2)
        return patch_inputs, patch_gt

    def get_patch_input(self, inputs, x1, y1, x2, y2):
        patch_inputs = inputs[y1:y2, x1:x2]
        return patch_inputs

    def get_patch_gt(self, x1, y1, x2, y2):
        gt = self.get_map(MapType.GT)
        patch_gt = gt[y1:y2, x1:x2]
        return patch_gt


def get_coords(stride_x, stride_y, patch_size_x, patch_size_y, x_max=8964, y_max=6716):
    return [(rmin, cmin, rmin+patch_size_x, cmin+patch_size_y) for rmin in range(0, x_max-patch_size_x + 1, stride_x)
            for cmin in range(0, y_max-patch_size_y + 1, stride_y)]
