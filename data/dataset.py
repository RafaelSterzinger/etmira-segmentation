from data.data_instance import get_coords
from config import SAMPLE_SIZE, get_slice
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import glob
import os


IMAGES_PER_PATCH = 10


class EtMirADatasetTraining(Dataset):
    def __init__(self, data_path=None, transform_input=None, transform_gt=None, use_augment=False, patch_sizes=512):
        super().__init__()
        self.image_paths = sorted(
            glob.glob(os.path.join(data_path, "*_inputs.npy")))
        self.gt_paths = list(
            map(lambda x: x.replace('inputs', 'gt'), self.image_paths))
        self.crop = A.Compose([A.CropNonEmptyMaskIfExists(
            patch_sizes, patch_sizes), A.Resize(SAMPLE_SIZE, SAMPLE_SIZE)])
        self.augment = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate()
        ]) if use_augment else None
        assert len(self.image_paths) == len(self.gt_paths)
        self.transform_input = transform_input
        self.transform_gt = transform_gt
        self.length = len(self.image_paths)*IMAGES_PER_PATCH

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx//IMAGES_PER_PATCH
        image_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        image = np.load(image_path)[:, :, get_slice()]

        gt = np.load(gt_path)
        cropped = self.crop(image=image, mask=gt)
        image = cropped['image']
        gt = cropped['mask']

        if self.augment:
            transformed = self.augment(image=image, mask=gt)
            image = transformed['image']
            gt = transformed['mask']

        if self.transform_input:
            image = self.transform_input(image)
        if self.transform_gt:
            gt = self.transform_gt(gt)
        return image, gt


class EtMirADatasetValidation(Dataset):
    def __init__(self, data_path=None, transform_input=None, transform_gt=None, patch_size=512):
        super().__init__()
        self.image_paths = sorted(
            glob.glob(os.path.join(data_path, "*_inputs.npy")))
        self.gt_paths = list(
            map(lambda x: x.replace('inputs', 'gt'), self.image_paths))
        self.resize = A.Resize(SAMPLE_SIZE, SAMPLE_SIZE)
        assert len(self.image_paths) == len(self.gt_paths)
        self.transform_input = transform_input
        self.transform_gt = transform_gt
        self.coords = get_coords(
            patch_size, patch_size, patch_size, patch_size)
        self.length = len(self.image_paths) * len(self.coords)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_idx = idx // len(self.coords)
        coord = self.coords[idx % len(self.coords)]
        image_path = self.image_paths[image_idx]
        gt_path = self.gt_paths[image_idx]
        image = np.array(np.load(image_path, mmap_mode='r')
                         [coord[1]:coord[3], coord[0]:coord[2], get_slice()])
        gt = np.array(np.load(gt_path, mmap_mode='r')[
                      coord[1]:coord[3], coord[0]:coord[2]])

        resized = self.resize(image=image, mask=gt)
        image = resized['image']
        gt = resized['mask']

        if self.transform_input:
            image = self.transform_input(image)
        if self.transform_gt:
            gt = self.transform_gt(gt)
        return image, gt


class EtMirADatasetInference(Dataset):
    def __init__(self, data_instance=None, transform_input=None, patch_size=512):
        super().__init__()
        self.data_instance = data_instance
        self.transform_input = transform_input
        self.coords = get_coords(
            patch_size//2, patch_size//2, patch_size, patch_size)
        self.inputs = self.data_instance.get_inputs()[
            :, :, get_slice()]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords = self.coords[idx]
        # only use necessary channel
        image = self.data_instance.get_patch_input(self.inputs, *coords)
        if self.transform_input:
            image = self.transform_input(image)
        return coords, image
